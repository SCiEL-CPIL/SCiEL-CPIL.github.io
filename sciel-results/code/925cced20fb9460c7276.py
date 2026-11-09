import numpy as np

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers an unknown function using a polynomial basis model.
    The model is a linear combination of 10 polynomial basis functions
    spanning constants, linear, quadratic, and cubic terms in x and y.
    """
    # 10 polynomial basis functions:
    # 0: 1
    # 1: x
    # 2: y
    # 3: x^2
    # 4: y^2
    # 5: x*y
    # 6: x^3
    # 7: y^3
    # 8: x^2*y
    # 9: x*y^2
    
    # Compute basis functions
    b0 = np.ones_like(x)
    b1 = x
    b2 = y
    b3 = x**2
    b4 = y**2
    b5 = x * y
    b6 = x**3
    b7 = y**3
    b8 = x**2 * y
    b9 = x * y**2
    
    # Stack into matrix of shape (N, 10)
    B = np.column_stack([b0, b1, b2, b3, b4, b5, b6, b7, b8, b9])
    
    # Prediction is B @ params
    pred = B.dot(params)
    
    return pred
