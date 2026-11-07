"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Fits a cubic polynomial model:
    f(x, y) = p0 + p1*x + p2*y + p3*x^2 + p4*y^2 + p5*x*y 
               + p6*x^3 + p7*y^3 + p8*x^2*y + p9*x*y^2
    
    This basis can represent any smooth function on [-1, 1]^2 reasonably well,
    including linear, quadratic, and simple periodic functions (via Taylor series).
    """
    # Ensure inputs are arrays
    x = np.asarray(x)
    y = np.asarray(y)
    
    # Precompute powers
    x2 = x * x
    y2 = y * y
    x3 = x2 * x
    y3 = y2 * y
    
    # Basis functions
    b0 = np.ones_like(x)
    b1 = x
    b2 = y
    b3 = x2
    b4 = y2
    b5 = x * y
    b6 = x3
    b7 = y3
    b8 = x2 * y
    b9 = x * y2
    
    # Linear combination
    pred = (params[0] * b0 + 
            params[1] * b1 + 
            params[2] * b2 + 
            params[3] * b3 + 
            params[4] * b4 + 
            params[5] * b5 + 
            params[6] * b6 + 
            params[7] * b7 + 
            params[8] * b8 + 
            params[9] * b9)
            
    return pred
# EVOLVE-BLOCK-END
