"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Nonlinear oscillator model (Duffing + Van der Pol + Periodic Drive).
    
    Basis functions (10 params):
    0: 1
    1: x
    2: x^3
    3: v
    4: v^3
    5: x*v
    6: cos(t)
    7: sin(t)
    8: cos(2*t)
    9: sin(2*t)
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
# EVOLVE-BLOCK-END
