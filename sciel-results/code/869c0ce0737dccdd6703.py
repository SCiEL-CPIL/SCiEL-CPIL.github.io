import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    # x and y are 1D arrays of shape (N,)
    # params is 1D array of shape (10,)
    
    # Ensure x and y are 1D for broadcasting
    x = x.flatten()
    y = y.flatten()
    
    # Create column vectors for a bivariate cubic polynomial basis
    # Terms: 1, x, y, x^2, xy, y^2, x^3, x^2y, xy^2, y^3
    X1 = np.ones_like(x)
    X2 = x
    X3 = y
    X4 = x**2
    X5 = x * y
    X6 = y**2
    X7 = x**3
    X8 = x**2 * y
    X9 = x * y**2
    X10 = y**3
    
    # Stack into a matrix of shape (N, 10)
    features = np.column_stack((X1, X2, X3, X4, X5, X6, X7, X8, X9, X10))
    
    # Linear combination
    return features @ params
