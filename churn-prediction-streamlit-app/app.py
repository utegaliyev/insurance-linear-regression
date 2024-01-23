import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
import pickle
import plotly.express as px


# Настройка страницы
st.set_page_config(
    page_title="Insurance Prediction App",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Глобальные переменные
uploaded_file = None

# Создание переменных session state
if 'df_input' not in st.session_state:
    st.session_state['df_input'] = pd.DataFrame()

if 'df_predicted' not in st.session_state:
    st.session_state['df_predicted'] = pd.DataFrame()

if 'tab_selected' not in st.session_state:
    st.session_state['tab_selected'] = None

def reset_session_state():
    st.session_state['df_input'] = pd.DataFrame()
    st.session_state['df_predicted'] = pd.DataFrame()

# logistic regression model
model_file_path = 'models/insurance_model.sav'
model = pickle.load(open(model_file_path, 'rb'))

# Кэширование функции предсказания
@st.cache_data
def predict_churn(df_input):

    df_original = df_input.copy()

    df_original['sex'] = df_original['sex'].map({'male':1,'female':0})
    df_original['smoker'] = df_original['smoker'].map({'yes':1,'no':0})
    df_original['region'] = df_original['region'].map({'southwest':0,'southeast':1,'northwest':2,'northeast':3})

    y_pred = model.predict(df_original)
    df_original['charges'] = y_pred
    return df_original


@st.cache_data
def convert_df(df):
    # Функция для конвертации датафрейма в csv
    return df.to_csv(index=False).encode('utf-8')

# Sidebar section start
# Сайдбар блок
with st.sidebar:
    st.title('🗂 Ввод данных')
    
    tab1, tab2 = st.tabs(['📁 Данные из файла', '📝 Ввести вручную'])
    with tab1:
        # Вкладка с загрузкой файла, выбором порога и кнопкой предсказания (вкладка 1)
        uploaded_file = st.file_uploader("Выбрать CSV файл", type=['csv', 'xlsx'], on_change=reset_session_state)
        if uploaded_file is not None:
            prediction_button = st.button('Предсказать', type='primary', use_container_width=True, key='button1')
            st.session_state['df_input'] = pd.read_csv(uploaded_file)
            if prediction_button:
                # Предсказание и сохранение в session state
                st.session_state['df_predicted'] = predict_churn(st.session_state['df_input'])
                st.session_state['tab_selected'] = 'tab1'

    with tab2:
        # Вкладка с вводом данных вручную, c кнопкой предсказания (вкладка 2)
        age = st.text_input('Age', placeholder='20', help='Введите возраст')
        sex = st.selectbox( 'Пол', ('female', 'male'))
        bmi = st.text_input('BMI', placeholder='20', help='Введите BMI')
        children_count = st.text_input('Children', placeholder='1', help='Введите кол-во детей')
        smoker = st.selectbox( 'Курильщик', ('yes', 'no'))
        region = st.selectbox('Регион', ('southwest', 'southeast', 'northwest', 'northeast'))
        
        # Если введен ID клиента, то показываем слайдер с порогом и кнопку предсказания
        if age != '':
            prediction_button_tab2 = st.button('Предсказать', type='primary', use_container_width=True, key='button2')
            
            if prediction_button_tab2:
                st.session_state['tab_selected'] = 'tab2'
                # Сохраняем введенные данные в session state в виде датафрейма
                #age,sex,bmi,children,smoker,region
                st.session_state['df_input'] = pd.DataFrame({
                    'age': age,
                    'sex': sex,
                    'bmi': bmi,
                    'children': children_count,
                    'smoker': smoker,
                    'region': region,
                }, index=[0])
                # Предсказание и сохранение в session state
                st.session_state['df_predicted'] = predict_churn(st.session_state['df_input'])

                

# Sidebar section end

# Main section start
# Основной блок
st.image('https://miro.medium.com/v2/resize:fit:720/format:webp/0*t-YEeLunFUQy12oa.png', width=400)
st.title('Прогнозирование суммы страховки')

with st.expander("Описание проекта"):
    st.write("""В данном проекте мы рассмотрим задачу прогнозирования суммы страховки.
    Для этого мы будем использовать датасет из открытых источников.
    Датасет содержит информацию о предыдущих клиентах, которые оплатили страховку.
    Наша задача - построить модель, которая будет предсказывать сумму страховки мед услуг.""")

# Вывод входных данных (из файла или введенных пользователем)
if len(st.session_state['df_input']) > 0:
    # Если предсказание еще не было сделано, то выводим входные данные в общем виде
    if len(st.session_state['df_predicted']) == 0:
        st.subheader('Данные из файла')
        st.write(st.session_state['df_input'])
    else:
        # Если предсказание уже было сделано, то выводим входные данные в expander
        with st.expander("Входные данные"):
            st.write(st.session_state['df_input'])

# Выводим результаты предсказания для отдельного клиента (вкладка 2)
if len(st.session_state['df_predicted']) > 0 and st.session_state['tab_selected'] == 'tab2':
    st.subheader(f'Сумма страховки: ${round( st.session_state["df_predicted"]["charges"][0])}')

# Выводим результаты предсказания для клинтов из файла (вкладка 1)
if len(st.session_state['df_predicted']) > 0 and st.session_state['tab_selected'] == 'tab1':
    # Результаты предсказания для всех клиентов в файле
    st.subheader('Результаты прогнозирования')
    st.write(st.session_state['df_predicted'])
    # Скачиваем результаты предсказания для всех клиентов в файле
    res_all_csv = convert_df(st.session_state['df_predicted'])
    st.download_button(
        label="Скачать все предсказания",
        data=res_all_csv,
        file_name='df-churn-predicted-all.csv',
        mime='text/csv',
    )

    # Гистограмма оттока для всех клиентов в файле
    fig = px.histplot(st.session_state['df_predicted'].charges,bins=20, kde=True,color='red')

    st.plotly_chart(fig, use_container_width=True)

    # Клиенты с высоким риском оттока
    predicted_charges = st.session_state['df_predicted']
    # Выводим клиентов с высоким риском оттока
    if len(predicted_charges) > 0:
        st.subheader('Клиенты с расчетом страховки')
        st.write(predicted_charges)
        # Скачиваем клиентов с расчетом страховки
        predicted_charges_csv = convert_df(predicted_charges)
        st.download_button(
            label="Скачать клиентов с расчетом страховки",
            data=predicted_charges_csv,
            file_name='df-predicted-charges.csv',
            mime='text/csv',
        )