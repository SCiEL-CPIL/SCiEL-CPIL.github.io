"""
State-space representation with time-varying coefficients derived from
a low-order Fourier series in t. Captures periodic changes in effective
stiffness and damping without high-degree polynomials.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Nonlinear oscillator acceleration with time-varying coefficients.
    
    Model: a(t) = k(t) * x + c(t) * v + f(t)
    where k(t), c(t), f(t) are low-order Fourier series in t.
    
    With 10 params:
    p0: bias for k(t)
    p1: cos(t) coefficient for k(t)
    p2: sin(t) coefficient for k(t)
    p3: bias for c(t)
    p4: cos(t) coefficient for c(t)
    p5: sin(t) coefficient for c(t)
    p6: bias for f(t)
    p7: cos(t) coefficient for f(t)
    p8: sin(t) coefficient for f(t)
    p9: x^3 coefficient (nonlinear stiffness)
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Precompute trig functions
    sin_t = np.sin(t)
    cos_t = np.cos(t)
    
    # Time-varying stiffness k(t)
    k_t = p0 + p1 * cos_t + p2 * sin_t
    
    # Time-varying damping c(t)
    c_t = p3 + p4 * cos_t + p5 * sin_t
    
    # Time-varying forcing f(t)
    f_t = p6 + p7 * cos_t + p8 * sin_t
    
    # Nonlinear term
    x3 = x * x * x
    
    # Construct the result
    result = k_t * x + c_t * v + f_t + p9 * x3
    
    return result
# EVOLVE-BLOCK-END
