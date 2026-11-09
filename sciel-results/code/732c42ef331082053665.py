"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression using a rich set of basis functions.
    
    The function uses a combination of:
    - Polynomial terms (up to degree 5)
    - Trigonometric terms (sin, cos at various frequencies)
    - Exponential terms (Gaussian-like)
    - Hyperbolic terms (tanh)
    
    This provides a flexible model that can approximate many functional forms.
    """
    # Extract parameters
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Compute basis functions
    x2 = x * x
    x3 = x2 * x
    x4 = x2 * x2
    x5 = x4 * x
    
    # Trigonometric terms
    sin_x = np.sin(x)
    cos_x = np.cos(x)
    sin_2x = np.sin(2 * x)
    cos_2x = np.cos(2 * x)
    
    # Exponential/Gaussian terms
    exp_neg_x2 = np.exp(-x2)
    exp_neg_x = np.exp(-np.abs(x))
    
    # Hyperbolic tangent
    tanh_x = np.tanh(x)
    
    # Combine basis functions with parameters
    result = (p0 * np.ones_like(x) + 
              p1 * x + 
              p2 * x2 + 
              p3 * x3 + 
              p4 * sin_x + 
              p5 * cos_x + 
              p6 * exp_neg_x2 + 
              p7 * tanh_x + 
              p8 * sin_2x + 
              p9 * cos_2x)
    
    return result
# EVOLVE-BLOCK-END
