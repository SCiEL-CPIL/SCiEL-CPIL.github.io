"""Symbolic regression with polynomial basis up to degree 3."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Predict y from x using a polynomial basis up to degree 3.
    Basis: 1, x, y, x^2, xy, y^2, x^3, x^2*y, x*y^2, y^3
    """
    p = params
    return (p[0] + 
            p[1] * x + 
            p[2] * y + 
            p[3] * x**2 + 
            p[4] * x * y + 
            p[5] * y**2 + 
            p[6] * x**3 + 
            p[7] * x**2 * y + 
            p[8] * x * y**2 + 
            p[9] * y**3)
# EVOLVE-BLOCK-END
