
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


np.random.seed(42)
num_samples = 100
features = np.random.rand(num_samples, 2)  
target = 100 + 50 * features[:, 0] + 75 * features[:, 1] + np.random.normal(0, 20, num_samples)  


data = pd.DataFrame({'Feature1': features[:, 0], 'Feature2': features[:, 1], 'Sales': target})


X = data[['Feature1', 'Feature2']]
y = data['Sales']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = LinearRegression()


model.fit(X_train, y_train)


predictions = model.predict(X_test)


mse = mean_squared_error(y_test, predictions)
print(f"Mean Squared Error: {mse}")


new_data = np.array([[0.6, 0.8]])  
predicted_sale = model.predict(new_data)
print(f"Predicted Diwali Sale: {predicted_sale[0]}")
