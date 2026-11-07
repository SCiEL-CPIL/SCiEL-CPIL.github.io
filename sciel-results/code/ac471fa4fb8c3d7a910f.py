"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Improved Van der Pol style nonlinear oscillator model.
    
    Refines the basis to emphasize mixed position-velocity interaction terms
    which are critical for limit-cycle stability and amplitude-dependent damping.
    
    Basis: {1, x, v, x^3, x^2*v, x*v, x*v^2, x^2*v^2, x^4, v^3}
    
    This design prioritizes:
    - Core VdP terms: x, v, x^2*v
    - Nonlinear stiffness: x^3, x^4
    - Interaction drag/damping: x*v, x*v^2, x^2*v^2
    - Cubic drag: v^3
    
    By replacing generic pure-high-order terms (like x^5) with coupled terms
    (like x^2*v^2), the model better captures the energy exchange dynamics
    of the oscillator within the training range.
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Precompute powers
    x2 = x * x
    x3 = x2 * x
    x4 = x2 * x2
    
    v2 = v * v
    v3 = v2 * v
    
    # Mixed terms
    xv = x * v
    x2v = x2 * v
    xv2 = x * v2
    x2v2 = x2 * v2
    
    # Linear combination
    return (p0 + 
            p1 * x + 
            p2 * v + 
            p3 * x3 + 
            p4 * x2v + 
            p5 * xv + 
            p6 * xv2 + 
            p7 * x2v2 + 
            p8 * x4 + 
            p9 * v3)
# EVOLVE-BLOCK-END
