"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a nonlinear oscillator using bounded nonlinearities.
    Uses tanh for smooth, bounded nonlinear terms in position and velocity,
    plus quadratic and cross terms.
    
    params: 10 fitted constants
    p0: constant offset
    p1: linear position coefficient
    p2: linear velocity coefficient  
    p3: amplitude of tanh position nonlinearity
    p4: scale of tanh position nonlinearity
    p5: amplitude of tanh velocity nonlinearity
    p6: scale of tanh velocity nonlinearity
    p7: cross term x*v
    p8: quadratic position term
    p9: quadratic velocity term
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Bounded nonlinear terms using tanh
    tanh_x = np.tanh(p4 * x)
    tanh_v = np.tanh(p6 * v)
    
    # Quadratic terms
    x2 = x * x
    v2 = v * v
    xv = x * v
    
    return (p0 + 
            p1 * x + 
            p2 * v + 
            p3 * tanh_x + 
            p5 * tanh_v + 
            p7 * xv + 
            p8 * x2 + 
            p9 * v2)
# EVOLVE-BLOCK-END
