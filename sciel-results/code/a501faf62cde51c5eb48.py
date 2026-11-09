"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a damped nonlinear oscillator with time-dependent driving force.
    Uses normalized features to improve BFGS convergence from initial guess of 1s.
    """
    # Normalize inputs to roughly [-1, 1] or [0, 1] scale to help BFGS
    # Ranges from prompt:
    # t: [30, 50] -> center 40, scale 10
    # x: [-0.2056, 0.1749] -> center -0.015, scale 0.2
    # v: [-0.3391, 0.3624] -> center 0.01, scale 0.35
    
    t_n = (t - 40.0) / 10.0
    x_n = (x + 0.015) / 0.2
    v_n = (v - 0.01) / 0.35
    
    # Compute features
    x2 = x_n * x_n
    x3 = x2 * x_n
    v2 = v_n * v_n
    xv = x_n * v_n
    
    sin_t = np.sin(t)
    cos_t = np.cos(t)
    
    # Combine features with params
    # p0: bias
    # p1: linear x
    # p2: linear v
    # p3: nonlinear x^3 (Duffing)
    # p4: drive sin
    # p5: drive cos
    # p6: quadratic x
    # p7: quadratic v
    # p8: coupling xv
    # p9: time drift
    
    return (params[0] + 
            params[1] * x_n + 
            params[2] * v_n + 
            params[3] * x3 + 
            params[4] * sin_t + 
            params[5] * cos_t + 
            params[6] * x2 + 
            params[7] * v2 + 
            params[8] * xv + 
            params[9] * t_n)
# EVOLVE-BLOCK-END
