"""
Symbolic regression with a mixed polynomial and trigonometric basis.
The model is linear in the parameters, which allows BFGS to efficiently
find the global minimum for the least squares objective.
"""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Computes the model prediction using a 10-term basis:
    p0 * 1 + p1 * x + p2 * x^2 + p3 * x^3 + p4 * x^4 +
    p5 * sin(x) + p6 * cos(x) + p7 * sin(2x) + p8 * cos(2x) +
    p9 * exp(-x^2)
    """
    # Ensure x is a numpy array
    x = np.asarray(x, dtype=float)
    
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
    
    # Combine linearly
    result = (p0 * 1.0 + 
              p1 * x + 
              p2 * x2 + 
              p3 * x3 + 
              p4 * x4 + 
              p5 * sin_x + 
              p6 * cos_x + 
              p7 * sin_2x + 
              p8 * cos_2x + 
              p9 * exp_neg_x2)
              
    return result
# EVOLVE-BLOCK-END
