"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Acceleration of a damped nonlinear oscillator with time-dependent driving force.
    
    Uses a rich basis of physically-motivated nonlinear terms:
    - Linear terms: t, x, v, constant
    - Nonlinear restoring: x^2, x^3
    - Nonlinear damping: x*v, v^2, x^2*v
    - Periodic driving: sin(t), cos(t)
    """
    # Extract parameters
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Compute basis functions
    t_term = t
    x_term = x
    v_term = v
    const_term = np.ones_like(x)
    x2_term = x ** 2
    x3_term = x ** 3
    xv_term = x * v
    v2_term = v ** 2
    x2v_term = x ** 2 * v
    sin_t = np.sin(t)
    cos_t = np.cos(t)
    
    # We have 10 params, so we select 10 basis terms
    # Combine linear and key nonlinear terms
    return (p0 * t_term + 
            p1 * x_term + 
            p2 * v_term + 
            p3 * const_term + 
            p4 * x2_term + 
            p5 * x3_term + 
            p6 * xv_term + 
            p7 * v2_term + 
            p8 * sin_t + 
            p9 * cos_t)
# EVOLVE-BLOCK-END
