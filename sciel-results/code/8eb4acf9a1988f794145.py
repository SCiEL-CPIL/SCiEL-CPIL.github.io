"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symmetric polynomial basis with 10 coefficients.
    Basis: [1, x, x^2, x^3, x^4, x^5, x^6, x^7, x^8, x^9]
    This allows fitting any smooth function well within [-2, 2].
    """
    # Ensure x is a numpy array
    x = np.asarray(x, dtype=float)
    
    # Extract parameters, ensuring finite values
    p0 = params[0]
    p1 = params[1]
    p2 = params[2]
    p3 = params[3]
    p4 = params[4]
    p5 = params[5]
    p6 = params[6]
    p7 = params[7]
    p8 = params[8]
    p9 = params[9]
    
    # Compute powers efficiently using Horner's method or direct powers
    # Direct powers are fine for degree 9
    x2 = x * x
    x3 = x2 * x
    x4 = x2 * x2
    x5 = x4 * x
    x6 = x3 * x3
    x7 = x6 * x
    x8 = x4 * x4
    x9 = x8 * x
    
    result = (p0 + p1 * x + p2 * x2 + p3 * x3 + p4 * x4 + 
              p5 * x5 + p6 * x6 + p7 * x7 + p8 * x8 + p9 * x9)
    
    return result
# EVOLVE-BLOCK-END
