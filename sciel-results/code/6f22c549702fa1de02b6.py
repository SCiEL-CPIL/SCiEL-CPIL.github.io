"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a nonlinear oscillator using a physically-motivated basis.
    
    Terms (10 params):
    p0: constant (driving offset)
    p1: x (linear restoring)
    p2: x^2 (quadratic restoring)
    p3: x^3 (cubic restoring, Duffing)
    p4: v (linear damping)
    p5: v^2 (quadratic damping)
    p6: v^3 (cubic damping)
    p7: x*v (coupling)
    p8: x^2*v (Van der Pol type nonlinear damping)
    p9: sin(x) (bounded nonlinearity for conditioning)
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    x2 = x * x
    x3 = x2 * x
    v2 = v * v
    v3 = v2 * v
    
    return (p0 
            + p1 * x 
            + p2 * x2 
            + p3 * x3 
            + p4 * v 
            + p5 * v2 
            + p6 * v3 
            + p7 * x * v 
            + p8 * x2 * v 
            + p9 * np.sin(x))
# EVOLVE-BLOCK-END
