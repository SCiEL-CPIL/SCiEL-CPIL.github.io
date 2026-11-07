# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    10-parameter model for symbolic regression with 3 inputs.
    
    Basis functions:
    0: 1 (constant)
    1: x
    2: y
    3: z
    4: x*y
    5: x*z
    6: y*z
    7: x*y*z
    8: x / (1 + y + z)
    9: sqrt(x + y + z)
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Ensure numerical stability for the rational and sqrt terms
    denom = 1.0 + y + z
    # Clamp to avoid division by zero (denom >= 1.5 since y,z >= 0.5)
    # Actually y,z in [0.5, 2], so denom in [2, 5], always positive
    rational_term = x / denom
    
    # sqrt term: x+y+z in [1.5, 6], always positive
    sqrt_term = np.sqrt(x + y + z)
    
    pred = (
        p0
        + p1 * x
        + p2 * y
        + p3 * z
        + p4 * x * y
        + p5 * x * z
        + p6 * y * z
        + p7 * x * y * z
        + p8 * rational_term
        + p9 * sqrt_term
    )
    
    return pred
# EVOLVE-BLOCK-END
