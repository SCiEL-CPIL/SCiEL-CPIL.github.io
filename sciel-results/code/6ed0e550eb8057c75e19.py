"""
Rewritten program for symbolic regression of a nonlinear oscillator.
Uses a physically-motivated basis including Duffing and Van der Pol terms,
and sinusoidal driving forces.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a nonlinear oscillator.
    Model: a = p0 + p1*x + p2*x^2 + p3*x^3 + p4*v + p5*v^3 + p6*x*v + p7*sin(t) + p8*cos(t) + p9*t
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Precompute powers and trig functions
    x2 = x * x
    x3 = x2 * x
    v2 = v * v
    v3 = v2 * v
    sin_t = np.sin(t)
    cos_t = np.cos(t)
    
    # Construct the linear combination of features
    # p0: bias
    # p1: linear x
    # p2: quadratic x
    # p3: cubic x (Duffing)
    # p4: linear v
    # p5: cubic v (Van der Pol / drag)
    # p6: x*v coupling
    # p7: sin(t) driving
    # p8: cos(t) driving
    # p9: linear t (drift)
    
    result = (p0 
              + p1 * x 
              + p2 * x2 
              + p3 * x3 
              + p4 * v 
              + p5 * v3 
              + p6 * x * v 
              + p7 * sin_t 
              + p8 * cos_t 
              + p9 * t)
              
    return result
# EVOLVE-BLOCK-END
