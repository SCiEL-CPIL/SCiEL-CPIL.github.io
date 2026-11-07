"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    p = params
    # Basis functions
    # 1, x, x^2, x^3, x^4, sin(x), cos(x), sin(2x), cos(2x), tanh(x)
    
    # Polynomial terms
    x2 = x * x
    x3 = x2 * x
    x4 = x2 * x2
    
    # Trigonometric terms
    sin_x = np.sin(x)
    cos_x = np.cos(x)
    sin_2x = np.sin(2.0 * x)
    cos_2x = np.cos(2.0 * x)
    
    # Non-linear smooth term
    tanh_x = np.tanh(x)
    
    return (p[0] + 
            p[1] * x + 
            p[2] * x2 + 
            p[3] * x3 + 
            p[4] * x4 + 
            p[5] * sin_x + 
            p[6] * cos_x + 
            p[7] * sin_2x + 
            p[8] * cos_2x + 
            p[9] * tanh_x)
# EVOLVE-BLOCK-END
