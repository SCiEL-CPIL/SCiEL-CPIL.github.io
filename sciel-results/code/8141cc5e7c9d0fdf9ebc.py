import numpy as np
import time

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a nonlinear oscillator (Duffing-like) with state-dependent damping.
    
    Model: a = p0 + p1*x + p2*v + p3*x^3 + p4*x*v + p5*x^2*v + p6*v^3 + p7*x^2 + p8*v^2 + p9*x^4
    
    This uses a complete polynomial basis up to 4th order to capture nonlinear stiffness (x^3, x^4)
    and nonlinear damping (v^3, x*v, x^2*v) terms typical in driven Duffing oscillators.
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Compute terms
    term0 = p0
    term1 = p1 * x
    term2 = p2 * v
    term3 = p3 * x**3
    term4 = p4 * x * v
    term5 = p5 * x**2 * v
    term6 = p6 * v**3
    term7 = p7 * x**2
    term8 = p8 * v**2
    term9 = p9 * x**4
    
    return term0 + term1 + term2 + term3 + term4 + term5 + term6 + term7 + term8 + term9
