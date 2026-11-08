import numpy as np
from scipy.optimize import minimize
import time

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Models acceleration for a nonlinear oscillator using a Duffing-Van der Pol
    structure with additional cross-terms. The structure is:
    
    a = c1*x + c2*x^3 + c3*v + c4*v^3 + c5*x^2*v + c6*x*v^2 + c7*x^2 + c8*v^2 + c9*x*v + c10
    
    where the coefficients are linear combinations of the 10 parameters to
    allow flexible fitting while maintaining a physically motivated structure.
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Precompute powers
    x2 = x * x
    v2 = v * v
    x3 = x2 * x
    v3 = v2 * v
    
    # Construct the model with a mix of physically motivated terms
    # Stiffness terms: linear and cubic in x
    # Damping terms: linear and cubic in v
    # Cross terms: x^2*v, x*v^2, x*v
    # Quadratic terms: x^2, v^2 (for asymmetry)
    # Constant term
    
    return (p0 +
            p1 * x +
            p2 * x3 +
            p3 * v +
            p4 * v3 +
            p5 * x2 * v +
            p6 * x * v2 +
            p7 * x2 +
            p8 * v2 +
            p9 * x * v)
