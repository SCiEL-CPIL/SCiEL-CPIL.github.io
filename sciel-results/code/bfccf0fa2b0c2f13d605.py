"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Strategy: Use a complete quadratic polynomial basis in raw x, y, z.
    
    Basis terms (10 parameters):
    0: 1
    1: x
    2: y
    3: z
    4: x^2
    5: y^2
    6: z^2
    7: x*y
    8: x*z
    9: y*z
    
    This model class captures additive polynomial structures up to second order.
    Observations from similar runs suggest the target function may be well-approximated
    by a low-degree polynomial in the raw inputs rather than a power law (log-space).
    """
    p = params
    
    return (p[0] + 
            p[1] * x + 
            p[2] * y + 
            p[3] * z + 
            p[4] * x * x + 
            p[5] * y * y + 
            p[6] * z * z + 
            p[7] * x * y + 
            p[8] * x * z + 
            p[9] * y * z)
# EVOLVE-BLOCK-END
