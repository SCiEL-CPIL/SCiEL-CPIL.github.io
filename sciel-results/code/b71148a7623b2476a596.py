"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers an unknown function using a linear combination of basis functions.
    Basis: 1, x, x^2, x^3, x^4, sin(x), cos(x), sin(2x), cos(2x), exp(-x^2/2)
    """
    # Ensure x is a numpy array
    x = np.asarray(x)
    
    # Precompute basis functions
    # p[0] * 1
    # p[1] * x
    # p[2] * x^2
    # p[3] * x^3
    # p[4] * x^4
    # p[5] * x^5
    # p[6] * tanh(x)
    # p[7] * sin(x)
    # p[8] * cos(x)
    # p[9] * exp(-x^2/2)
    
    x2 = x * x
    x3 = x2 * x
    x4 = x2 * x2
    x5 = x4 * x
    
    tanh_x = np.tanh(x)
    sin_x = np.sin(x)
    cos_x = np.cos(x)
    exp_term = np.exp(-0.5 * x2)
    
    y = (params[0] * 1.0 +
         params[1] * x +
         params[2] * x2 +
         params[3] * x3 +
         params[4] * x4 +
         params[5] * x5 +
         params[6] * tanh_x +
         params[7] * sin_x +
         params[8] * cos_x +
         params[9] * exp_term)
         
    return y
# EVOLVE-BLOCK-END
