"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Approximate the hidden function using a 9th-degree polynomial.
    The input x is scaled to [-1, 1] to improve numerical stability for BFGS.
    params: array of length 10 [p0, p1, ..., p9]
    """
    # Scale x from [-2, 2] to [-1, 1]
    t = 0.5 * x
    
    # Construct the polynomial: p0 + p1*t + p2*t^2 + ... + p9*t^9
    # Using Horner's method or direct power calculation.
    # Direct power is fine for degree 9.
    
    # To be safe against potential non-finite values if params explode, 
    # we could clip, but BFGS usually keeps params in check for polynomials.
    
    y = np.zeros_like(x, dtype=float)
    
    # p0
    y += params[0]
    # p1 * t
    y += params[1] * t
    # p2 * t^2
    t2 = t * t
    y += params[2] * t2
    # p3 * t^3
    t3 = t2 * t
    y += params[3] * t3
    # p4 * t^4
    t4 = t2 * t2
    y += params[4] * t4
    # p5 * t^5
    t5 = t4 * t
    y += params[5] * t5
    # p6 * t^6
    t6 = t3 * t3
    y += params[6] * t6
    # p7 * t^7
    t7 = t6 * t
    y += params[7] * t7
    # p8 * t^8
    t8 = t4 * t4
    y += params[8] * t8
    # p9 * t^9
    t9 = t8 * t
    y += params[9] * t9
    
    return y
# EVOLVE-BLOCK-END
