"""Symbolic regression: nonlinear oscillator acceleration."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form model for the acceleration of a nonlinear oscillator.
    
    Structural Change:
    1. Internal normalization of inputs to improve conditioning for BFGS.
       t is centered at 40 and scaled by 10.
       x is scaled by 0.2.
       v is scaled by 0.35.
    2. Basis functions are chosen to be symmetric polynomials in the normalized
       state variables (x, v) and linear/trigonometric in time.
       This avoids the ill-conditioning caused by large 't' values interacting
       with small 'x/v' values in raw space.
       
    Model:
      a = p0 
        + p1 * t_norm
        + p2 * x_norm
        + p3 * v_norm
        + p4 * x_norm^2
        + p5 * v_norm^2
        + p6 * x_norm * v_norm
        + p7 * x_norm^3
        + p8 * v_norm^3
        + p9 * sin(t_norm)
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params

    # Normalization to stabilize optimization
    # t in [30, 50] -> center 40, range 10 -> [-1, 1]
    t_norm = (t - 40.0) / 10.0
    
    # x in [-0.2056, 0.1749] -> scale 0.2 -> approx [-1, 1]
    x_norm = x / 0.2
    
    # v in [-0.3391, 0.3624] -> scale 0.35 -> approx [-1, 1]
    v_norm = v / 0.35

    # Precompute powers and products
    x2 = x_norm * x_norm
    x3 = x2 * x_norm
    v2 = v_norm * v_norm
    v3 = v2 * v_norm
    xv = x_norm * v_norm

    # Closed-form expression
    # p0: offset
    # p1: linear time drift
    # p2: linear position restoring
    # p3: linear velocity damping
    # p4: quadratic position correction
    # p5: quadratic velocity correction
    # p6: position-velocity coupling
    # p7: cubic position (Duffing-like)
    # p8: cubic velocity (drag-like)
    # p9: periodic forcing
    result = (p0 
              + p1 * t_norm 
              + p2 * x_norm 
              + p3 * v_norm 
              + p4 * x2 
              + p5 * v2 
              + p6 * xv 
              + p7 * x3 
              + p8 * v3 
              + p9 * np.sin(t_norm))
    
    return result
# EVOLVE-BLOCK-END
