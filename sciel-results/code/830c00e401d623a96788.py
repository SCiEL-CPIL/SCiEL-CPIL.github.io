"""Fourier series expansion for symbolic regression."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form expression using a Fourier series basis in x and y.
    params: array of length 10, used as coefficients for the basis functions.
    
    Basis functions (10 total):
    0: 1 (constant)
    1: cos(x)
    2: sin(x)
    3: cos(y)
    4: sin(y)
    5: cos(x + y)
    6: sin(x + y)
    7: cos(x - y)
    8: sin(x - y)
    9: cos(2*x) * cos(2*y)  -- a higher-order interaction term
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Precompute trig arguments
    cx = np.cos(x)
    sx = np.sin(x)
    cy = np.cos(y)
    sy = np.sin(y)
    
    cxy_plus = np.cos(x + y)
    sxy_plus = np.sin(x + y)
    cxy_minus = np.cos(x - y)
    sxy_minus = np.sin(x - y)
    c2x_c2y = np.cos(2 * x) * np.cos(2 * y)
    
    # Combine with parameters
    result = (
        p0
        + p1 * cx
        + p2 * sx
        + p3 * cy
        + p4 * sy
        + p5 * cxy_plus
        + p6 * sxy_plus
        + p7 * cxy_minus
        + p8 * sxy_minus
        + p9 * c2x_c2y
    )
    
    return result
# EVOLVE-BLOCK-END
