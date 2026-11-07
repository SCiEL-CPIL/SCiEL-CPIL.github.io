"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symmetric Van der Pol style nonlinear oscillator model.
    
    Uses a basis of odd-parity terms to capture limit-cycle dynamics while
    respecting the symmetry of the oscillator (f(-x, -v) = -f(x, v)).
    Basis: {x, v, x^3, x^2*v, v^3, x^5, x^4*v, x*v^2, x^3*v^2, x^2*v^3}
    
    This prioritizes:
    - Core VdP terms: x, v, x^3, x^2*v, v^3
    - Higher order stiffness: x^5
    - Higher order damping: x^4*v, x*v^2, x^3*v^2, x^2*v^3
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    x2 = x * x
    x3 = x2 * x
    x4 = x2 * x2
    x5 = x4 * x
    
    v2 = v * v
    v3 = v2 * v
    
    x2v = x2 * v
    x4v = x4 * v
    xv2 = x * v2
    x3v2 = x3 * v2
    x2v3 = x2 * v3
    
    return (p0 * x + 
            p1 * v + 
            p2 * x3 + 
            p3 * x2v + 
            p4 * v3 + 
            p5 * x5 + 
            p6 * x4v + 
            p7 * xv2 + 
            p8 * x3v2 + 
            p9 * x2v3)
# EVOLVE-BLOCK-END
