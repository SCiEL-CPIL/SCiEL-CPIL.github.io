"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Evaluates the model using a rich basis of polynomial, trigonometric,
    hyperbolic, and Gaussian functions.
    params[0..9] are coefficients for:
    1, x, x^2, x^3, sin(x), cos(x), sin(2x), cos(2x), tanh(x), exp(-x^2)
    """
    # Ensure x is a numpy array
    x = np.asarray(x, dtype=np.float64)
    
    # Extract parameters
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Compute basis functions
    x2 = x * x
    x3 = x2 * x
    
    sin_x = np.sin(x)
    cos_x = np.cos(x)
    sin_2x = np.sin(2.0 * x)
    cos_2x = np.cos(2.0 * x)
    tanh_x = np.tanh(x)
    exp_neg_x2 = np.exp(-x2)
    
    # Combine
    y = (p0 * 1.0 +
         p1 * x +
         p2 * x2 +
         p3 * x3 +
         p4 * sin_x +
         p5 * cos_x +
         p6 * sin_2x +
         p7 * cos_2x +
         p8 * tanh_x +
         p9 * exp_neg_x2)
         
    return y
# EVOLVE-BLOCK-END
