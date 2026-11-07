"""Symbolic regression: multi-component model with robust fitting."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    A flexible model: combination of polynomial, exponential, and rational terms.
    params[0..9] are coefficients.
    
    Model:
    y = p0*x^3 + p1*x^2 + p2*x + p3 
      + p4*exp(p5*x) 
      + p6*sin(p7*x) 
      + p8*cos(p9*x)
    
    This gives 10 parameters covering a wide range of functional forms.
    """
    p = params
    x = np.asarray(x, dtype=float)
    
    # Polynomial part (cubic)
    poly = p[0] * x**3 + p[1] * x**2 + p[2] * x + p[3]
    
    # Exponential part
    exp_term = p[4] * np.exp(np.clip(p[5] * x, -50, 50))
    
    # Trigonometric parts
    sin_term = p[6] * np.sin(p[7] * x)
    cos_term = p[8] * np.cos(p[9] * x)
    
    return poly + exp_term + sin_term + cos_term

# EVOLVE-BLOCK-END
