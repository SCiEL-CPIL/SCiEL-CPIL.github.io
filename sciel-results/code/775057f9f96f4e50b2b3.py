import numpy as np

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Use a mixed basis of polynomial and trigonometric functions to capture
    # both local polynomial behavior and periodic/oscillatory structures.
    # Basis functions:
    # 0: 1 (intercept)
    # 1: x
    # 2: y
    # 3: x^2
    # 4: y^2
    # 5: x*y
    # 6: sin(x)
    # 7: sin(y)
    # 8: cos(x)
    # 9: cos(y)
    
    pred = (params[0] * np.ones_like(x) + 
            params[1] * x + 
            params[2] * y + 
            params[3] * x * x + 
            params[4] * y * y + 
            params[5] * x * y + 
            params[6] * np.sin(x) + 
            params[7] * np.sin(y) + 
            params[8] * np.cos(x) + 
            params[9] * np.cos(y))
            
    return pred
