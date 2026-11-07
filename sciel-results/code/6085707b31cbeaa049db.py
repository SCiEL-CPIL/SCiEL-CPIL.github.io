"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form model for a damped nonlinear oscillator with time-dependent driving.
    
    Model: a = c0*t + c1*x + c2*v + c3*x^2 + c4*v^2 + c5*x*v + c6*sin(x) + c7*cos(t) + c8*x^3 + c9
    This captures linear dynamics, quadratic/nonlinear restoring and damping,
    trigonometric forcing, and cubic stiffness.
    """
    p = params
    # Ensure we have at least 10 params; pad if necessary
    if len(p) < 10:
        p = np.concatenate([p, np.ones(10 - len(p))])
    
    t = np.asarray(t, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    
    c0, c1, c2, c3, c4, c5, c6, c7, c8, c9 = p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9]
    
    # Compute terms
    term_t = c0 * t
    term_x = c1 * x
    term_v = c2 * v
    term_x2 = c3 * x * x
    term_v2 = c4 * v * v
    term_xv = c5 * x * v
    term_sinx = c6 * np.sin(x)
    term_cost = c7 * np.cos(t)
    term_x3 = c8 * x * x * x
    term_const = c9
    
    result = term_t + term_x + term_v + term_x2 + term_v2 + term_xv + term_sinx + term_cost + term_x3 + term_const
    
    return result
# EVOLVE-BLOCK-END
