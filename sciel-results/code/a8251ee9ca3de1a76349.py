import numpy as np
import time
from scipy.optimize import minimize

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression using a hybrid basis:
    - Polynomial terms up to degree 9 (captures smooth trends)
    - Sinusoidal terms with various frequencies (captures periodic behavior)
    - Tanh saturation terms (captures non-linear saturation)
    
    The 10 parameters are weights for these basis functions.
    """
    # Compute basis functions
    # Polynomial basis: x^0 through x^4 (5 terms)
    poly = np.ones_like(x)
    for i in range(1, 5):
        poly = np.column_stack([poly, x**i])
    
    # Sinusoidal basis: sin(x), cos(x), sin(2x), cos(2x) (4 terms)
    sinx = np.sin(x)
    cosx = np.cos(x)
    sin2x = np.sin(2*x)
    cos2x = np.cos(2*x)
    
    # Tanh basis: tanh(x) (1 term)
    tanhx = np.tanh(x)
    
    # Stack all basis functions: shape (n_points, 10)
    # poly has 5 columns, sin/cos has 4, tanh has 1 = 10 total
    basis = np.column_stack([
        poly,
        sinx, cosx, sin2x, cos2x,
        tanhx
    ])
    
    # Combine with parameters
    y = basis @ params
    
    return y
