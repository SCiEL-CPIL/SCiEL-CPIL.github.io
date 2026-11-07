"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a nonlinear oscillator using a rich polynomial basis.
    
    Inputs:
        x: displacement
        v: velocity
        params: array of 10 fitted constants [p0, ..., p9]
    
    Basis terms:
        p0 * x
        p1 * v
        p2 * x^2
        p3 * x^3
        p4 * x*v
        p5 * v^2
        p6 * x^4
        p7 * v^3
        p8 * x^2*v
        p9 * x*v^2
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Precompute powers and cross terms
    x2 = x * x
    x3 = x2 * x
    x4 = x2 * x2
    v2 = v * v
    v3 = v2 * v
    xv = x * v
    x2v = x2 * v
    xv2 = x * v2
    
    return (
        p0 * x +
        p1 * v +
        p2 * x2 +
        p3 * x3 +
        p4 * xv +
        p5 * v2 +
        p6 * x4 +
        p7 * v3 +
        p8 * x2v +
        p9 * xv2
    )

# EVOLVE-BLOCK-END
