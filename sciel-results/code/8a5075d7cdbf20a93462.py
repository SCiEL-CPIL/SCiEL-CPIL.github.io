"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Non-linear parametric model with 10 free parameters:
      y = p0 + p1*x + p2*x^2 + p3*exp(-p4*x) + p5*sin(p6*x + p7) + p8/(1 + p9*x)
    
    This model allows for:
    - Quadratic polynomial background
    - Exponential decay with variable rate
    - Sinusoidal oscillation with variable frequency and phase
    - Rational term with variable pole location
    
    The BFGS optimizer will fit these 10 parameters.
    """
    x = np.asarray(x, dtype=float)
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Polynomial terms
    term_poly = p0 + p1 * x + p2 * x**2
    
    # Exponential term
    # Clip exponent to avoid overflow/underflow issues during optimization
    exp_arg = -p4 * x
    exp_arg = np.clip(exp_arg, -50, 50)
    term_exp = p3 * np.exp(exp_arg)
    
    # Sinusoidal term
    term_sin = p5 * np.sin(p6 * x + p7)
    
    # Rational term
    # Avoid division by zero
    denom = 1.0 + p9 * x
    # If denom is close to 0, add a small epsilon
    denom = np.where(np.abs(denom) < 1e-6, 1e-6, denom)
    term_rat = p8 / denom
    
    return term_poly + term_exp + term_sin + term_rat
# EVOLVE-BLOCK-END
