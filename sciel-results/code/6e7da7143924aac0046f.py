"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Higher-order polynomial basis in x and y with 10 parameters.
    Uses monomials up to degree 4 to better capture smooth nonlinearities.
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params

    x2 = x * x
    y2 = y * y
    xy = x * y

    x3 = x2 * x
    y3 = y2 * y
    x2y = x2 * y
    xy2 = x * y2

    return (p0 + p1 * x + p2 * y + p3 * x2 + p4 * y2 + p5 * xy +
            p6 * x3 + p7 * y3 + p8 * x2y + p9 * xy2)
# EVOLVE-BLOCK-END
