"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form expression for a driven nonlinear oscillator.
    
    Features (10 params):
    0: bias
    1: x (linear restoring)
    2: x^3 (Duffing nonlinearity)
    3: v (linear damping)
    4: v*|v| (quadratic damping)
    5: sin(t) (driving, sine)
    6: cos(t) (driving, cosine)
    7: sin(t)*x (parametric coupling)
    8: cos(t)*x (parametric coupling)
    9: x*v (nonlinear coupling)
    """
    p = params
    x2 = x * x
    x3 = x2 * x
    v2 = v * v
    v_abs = np.abs(v)
    v_vabs = v * v_abs
    sin_t = np.sin(t)
    cos_t = np.cos(t)
    sin_t_x = sin_t * x
    cos_t_x = cos_t * x
    xv = x * v
    
    return (p[0] +
            p[1] * x +
            p[2] * x3 +
            p[3] * v +
            p[4] * v_vabs +
            p[5] * sin_t +
            p[6] * cos_t +
            p[7] * sin_t_x +
            p[8] * cos_t_x +
            p[9] * xv)
# EVOLVE-BLOCK-END
