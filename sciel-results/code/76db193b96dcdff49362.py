"""Fourier series basis for symbolic regression."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Fourier series basis with 10 parameters:
    p0: constant (DC)
    p1: sin(x)
    p2: cos(x)
    p3: sin(2x)
    p4: cos(2x)
    p5: sin(3x)
    p6: cos(3x)
    p7: sin(4x)
    p8: cos(4x)
    p9: sin(5x)
    """
    p = params
    result = (
        p[0]
        + p[1] * np.sin(x)
        + p[2] * np.cos(x)
        + p[3] * np.sin(2 * x)
        + p[4] * np.cos(2 * x)
        + p[5] * np.sin(3 * x)
        + p[6] * np.cos(3 * x)
        + p[7] * np.sin(4 * x)
        + p[8] * np.cos(4 * x)
        + p[9] * np.sin(5 * x)
    )
    return result
# EVOLVE-BLOCK-END
