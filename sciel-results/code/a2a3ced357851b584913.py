"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import warnings


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Rich basis function for symbolic regression with 1 input.
    Uses a combination of polynomial, trigonometric, and smooth nonlinear basis functions.
    """
    p = params
    # Ensure finite params to avoid NaN/Inf
    if not np.all(np.isfinite(p)):
        p = np.ones(10)
    
    # Polynomial terms
    x2 = x ** 2
    x3 = x ** 3
    x4 = x ** 4
    x5 = x ** 5
    x6 = x ** 6
    
    term_poly = (p[0] * 1.0 + 
                 p[1] * x + 
                 p[2] * x2 + 
                 p[3] * x3 + 
                 p[4] * x4)
    
    # Higher-order polynomial terms (using remaining params)
    # We have 10 params total: p[0]..p[9]
    # Let's use:
    # p[0] = 1
    # p[1] = x
    # p[2] = x^2
    # p[3] = x^3
    # p[4] = x^4
    # p[5] = sin(x)
    # p[6] = cos(x)
    # p[7] = sin(2x)
    # p[8] = tanh(x)
    # p[9] = x * tanh(x)
    
    # Trigonometric terms
    sin_x = np.sin(x)
    cos_x = np.cos(x)
    sin_2x = np.sin(2.0 * x)
    
    term_trig = (p[5] * sin_x + 
                 p[6] * cos_x + 
                 p[7] * sin_2x)
    
    # Non-linear smooth terms
    tanh_x = np.tanh(x)
    term_nonlinear = (p[8] * tanh_x + 
                      p[9] * x * tanh_x)
    
    result = term_poly + term_trig + term_nonlinear
    
    # Safety: replace any non-finite values with 0
    result = np.where(np.isfinite(result), result, 0.0)
    
    return result
# EVOLVE-BLOCK-END
