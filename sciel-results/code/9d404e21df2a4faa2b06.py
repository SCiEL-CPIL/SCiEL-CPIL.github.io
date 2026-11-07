"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Evaluates the model using a basis of polynomial and trigonometric functions.
    params[0..9] are coefficients for:
    1, x, x^2, x^3, x^4, sin(x), cos(x), sin(2x), cos(2x), exp(-x^2)
    """
    # Ensure x is a numpy array
    x = np.asarray(x, dtype=np.float64)
    
    # Extract parameters
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Compute basis functions
    x2 = x * x
    x3 = x2 * x
    x4 = x2 * x2
    
    sin_x = np.sin(x)
    cos_x = np.cos(x)
    sin_2x = np.sin(2 * x)
    cos_2x = np.cos(2 * x)
    exp_neg_x2 = np.exp(-x2)
    
    # Combine
    y = (p0 * 1.0 +
         p1 * x +
         p2 * x2 +
         p3 * x3 +
         p4 * x4 +
         p5 * sin_x +
         p6 * cos_x +
         p7 * sin_2x +
         p8 * cos_2x +
         p9 * exp_neg_x2)
         
    return y
# EVOLVE-BLOCK-END
