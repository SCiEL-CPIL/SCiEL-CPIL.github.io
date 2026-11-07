"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Computes a general quadratic polynomial in x, y, z.
    Basis:
    0: 1
    1: x
    2: y
    3: z
    4: x^2
    5: y^2
    6: z^2
    7: xy
    8: xz
    9: yz
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
