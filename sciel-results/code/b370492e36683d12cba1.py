"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Rich basis: polynomials up to degree 5, trig, and hyperbolic
    p = params
    x2 = x * x
    x3 = x2 * x
    x4 = x2 * x2
    x5 = x4 * x
    
    # Basis functions
    b0 = np.ones_like(x)
    b1 = x
    b2 = x2
    b3 = x3
    b4 = x4
    b5 = x5
    b6 = np.sin(x)
    b7 = np.cos(x)
    b8 = np.sinh(x)
    b9 = np.cosh(x)
    
    # Linear combination
    y = (p[0] * b0 + 
         p[1] * b1 + 
         p[2] * b2 + 
         p[3] * b3 + 
         p[4] * b4 + 
         p[5] * b5 + 
         p[6] * b6 + 
         p[7] * b7 + 
         p[8] * b8 + 
         p[9] * b9)
    
    return y
# EVOLVE-BLOCK-END
