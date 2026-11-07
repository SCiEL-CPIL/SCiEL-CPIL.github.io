"""Symbolic regression: nonlinear oscillator (2) - closed-form equation."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a damped nonlinear oscillator with time-dependent driving force.
    
    Uses a scaled Duffing + Parametric Forcing basis:
    - Linear dynamics (damping, restoring)
    - Nonlinear stiffness (x^2, x^3)
    - Velocity nonlinearity (v^2)
    - Time-dependent driving (t, t^2)
    - Parametric cross terms (t*x, t*v)
    
    Inputs are scaled to O(1) to improve conditioning for BFGS.
    10 fitted constants.
    """
    # Scale inputs to approximately [-1, 1] range for numerical stability
    # t in [30, 50] -> center 40, scale 10
    T = (t - 40.0) / 10.0
    # x in [-0.2, 0.17] -> scale 0.2
    X = x / 0.2
    # v in [-0.34, 0.36] -> scale 0.35
    V = v / 0.35

    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params

    # Compute basis terms
    term_1 = 1.0
    term_2 = X
    term_3 = V
    term_4 = X**3
    term_5 = X**2
    term_6 = V**2
    term_7 = T
    term_8 = T**2
    term_9 = X * T
    term_10 = V * T

    # Linear combination
    result = (p0 * term_1 + p1 * term_2 + p2 * term_3 + p3 * term_4 + 
              p4 * term_5 + p5 * term_6 + p6 * term_7 + p7 * term_8 + 
              p8 * term_9 + p9 * term_10)
    
    return result
# EVOLVE-BLOCK-END
