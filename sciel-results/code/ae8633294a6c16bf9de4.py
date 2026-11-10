"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Linear-in-parameters model with a diverse basis of functions.
    
    y(x) = p[0]*1 + p[1]*x + p[2]*x^2 + p[3]*x^3 + p[4]*x^4 
           + p[5]*sin(x) + p[6]*cos(x) + p[7]*exp(-x^2) 
           + p[8]*tanh(x) + p[9]*x*exp(-x^2)
    
    This structure makes the optimization landscape convex (linear least squares),
    allowing BFGS to converge reliably to the global minimum, unlike the
    previous rational function model.
    """
    x = np.asarray(x, dtype=np.float64)
    p = params
    
    # Compute basis functions
    # Polynomial terms
    x2 = x * x
    x3 = x2 * x
    x4 = x2 * x2
    
    # Trigonometric terms
    sin_x = np.sin(x)
    cos_x = np.cos(x)
    
    # Exponential/Gaussian terms
    exp_neg_x2 = np.exp(-x2)
    
    # Hyperbolic tangent
    tanh_x = np.tanh(x)
    
    # Mixed term
    x_exp_neg_x2 = x * exp_neg_x2
    
    # Combine linearly
    # p[0] is constant term
    result = (p[0] * 1.0 + 
              p[1] * x + 
              p[2] * x2 + 
              p[3] * x3 + 
              p[4] * x4 + 
              p[5] * sin_x + 
              p[6] * cos_x + 
              p[7] * exp_neg_x2 + 
              p[8] * tanh_x + 
              p[9] * x_exp_neg_x2)
    
    # Ensure finite output
    result = np.where(np.isfinite(result), result, 0.0)
    
    return result
# EVOLVE-BLOCK-END
