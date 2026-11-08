"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

# The evaluator calls equation(t, x, v, params).
# Nonlinear oscillator acceleration model with coupled terms:
# y = driving(t) + spring(x) + damping(x, v) + parametric(t, x, v)
# with 10 fitted constants total.

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form expression for a nonlinear oscillator acceleration.

    Features chosen to capture:
    - Periodic driving: sin(t), cos(t)
    - Nonlinear spring: x, x^2, x^3 (Duffing)
    - State-dependent damping: v, x*v, x^2*v (Van der Pol)
    - Parametric coupling: sin(t)*x, cos(t)*v

    y = p0*sin(t) + p1*cos(t)
        + p2*x + p3*x^2 + p4*x^3
        + p5*v + p6*x*v + p7*x^2*v
        + p8*sin(t)*x + p9*cos(t)*v
    """
    sin_t = np.sin(t)
    cos_t = np.cos(t)

    x2 = x * x
    x3 = x2 * x
    xv = x * v
    x2v = x2 * v
    sin_t_x = sin_t * x
    cos_t_v = cos_t * v

    y = (params[0] * sin_t
         + params[1] * cos_t
         + params[2] * x
         + params[3] * x2
         + params[4] * x3
         + params[5] * v
         + params[6] * xv
         + params[7] * x2v
         + params[8] * sin_t_x
         + params[9] * cos_t_v)

    return y
# EVOLVE-BLOCK-END
