"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Acceleration of a damped nonlinear oscillator.
    
    Physics-structured basis: restoring, damping, nonlinear terms.
    No input normalization - direct physical quantities for BFGS stability.
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    x2 = x * x
    x3 = x2 * x
    v2 = v * v
    v3 = v2 * v
    xv = x * v
    x2v = x2 * v
    xv2 = x * v2
    
    return (p0 * x + 
            p1 * v + 
            p2 + 
            p3 * x3 + 
            p4 * xv + 
            p5 * v3 + 
            p6 * x2 + 
            p7 * x2v + 
            p8 * xv2 + 
            p9 * v2)
# EVOLVE-BLOCK-END
