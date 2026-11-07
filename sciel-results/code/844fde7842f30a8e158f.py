"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Returns a prediction using a complete polynomial basis of degree 3.
    Basis: 1, x, y, x^2, y^2, xy, x^3, y^3, x^2y, xy^2
    This allows the model to represent any polynomial of degree <= 3 exactly,
    and provides a strong local approximation for smooth non-polynomial functions
    on the domain [-1, 1].
    """
    # Extract parameters
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params

    # Compute basis terms
    # Using Horner-like or direct powers. Direct powers are fine for small degree.
    x2 = x * x
    y2 = y * y
    x3 = x2 * x
    y3 = y2 * y
    xy = x * y
    x2y = x2 * y
    xy2 = x * y2

    # Construct prediction
    # p0 * 1 + p1 * x + p2 * y + p3 * x^2 + p4 * y^2 + p5 * xy + p6 * x^3 + p7 * y^3 + p8 * x^2y + p9 * xy^2
    return p0 + p1 * x + p2 * y + p3 * x2 + p4 * y2 + p5 * xy + p6 * x3 + p7 * y3 + p8 * x2y + p9 * xy2
# EVOLVE-BLOCK-END
