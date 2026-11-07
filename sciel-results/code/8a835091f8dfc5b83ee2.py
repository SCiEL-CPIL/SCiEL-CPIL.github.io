# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Full quadratic polynomial basis.
    Basis functions:
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
    """
    p = params
    pred = (
        p[0] * 1.0 +
        p[1] * x +
        p[2] * y +
        p[3] * z +
        p[4] * x**2 +
        p[5] * y**2 +
        p[6] * z**2 +
        p[7] * x * y +
        p[8] * x * z +
        p[9] * y * z
    )
    return pred
# EVOLVE-BLOCK-END
