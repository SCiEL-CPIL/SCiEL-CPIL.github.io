import numpy as np

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Nonlinear oscillator model combining polynomial dynamics with periodic forcing.
    
    Basis functions (10 params):
    0: 1 (bias)
    1: x (linear spring)
    2: x^3 (Duffing nonlinearity)
    3: v (linear damping)
    4: v^3 (nonlinear damping)
    5: x*v (cross-coupling)
    6: cos(t) (periodic drive)
    7: sin(t) (periodic drive)
    8: cos(2*t) (second harmonic)
    9: sin(2*t) (second harmonic)
    """
    p = params
    
    # Polynomial terms
    term_x = p[1] * x
    term_x3 = p[2] * x**3
    term_v = p[3] * v
    term_v3 = p[4] * v**3
    term_xv = p[5] * x * v
    
    # Periodic driving terms
    ct = np.cos(t)
    st = np.sin(t)
    term_ct = p[6] * ct
    term_st = p[7] * st
    term_c2t = p[8] * np.cos(2 * t)
    term_s2t = p[9] * np.sin(2 * t)
    
    # Bias
    bias = p[0]
    
    return bias + term_x + term_x3 + term_v + term_v3 + term_xv + term_ct + term_st + term_c2t + term_s2t
