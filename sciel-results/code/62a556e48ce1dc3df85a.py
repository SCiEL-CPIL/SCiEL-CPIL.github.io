"""Rewritten program for symbolic regression with 1 input."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Flexible non-linear parametric model with 10 free parameters.
    
    y = p0 + p1*x + p2*x^2 + p3*x^3 
      + p4 * exp(-p5*x)
      + p6 * sin(p7*x + p8)
      + p9 * x^4
    
    This model covers:
    - Polynomial up to degree 4
    - Exponential decay with variable rate
    - Sinusoidal oscillation with variable frequency and phase
    
    The BFGS optimizer will fit these 10 parameters.
    """
    x = np.asarray(x, dtype=float)
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Polynomial terms up to degree 4
    term_poly = p0 + p1 * x + p2 * x**2 + p3 * x**3 + p9 * x**4
    
    # Exponential decay term with variable rate
    # Clip exponent to avoid overflow/underflow
    exp_arg = -p5 * x
    exp_arg = np.clip(exp_arg, -50, 50)
    term_exp = p4 * np.exp(exp_arg)
    
    # Sinusoidal term with variable frequency and phase
    term_sin = p6 * np.sin(p7 * x + p8)
    
    return term_poly + term_exp + term_sin
# EVOLVE-BLOCK-END
