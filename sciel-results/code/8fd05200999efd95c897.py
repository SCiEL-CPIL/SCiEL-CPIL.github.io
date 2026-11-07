"""Fourier series expansion for symbolic regression."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form expression using an expanded Fourier series basis in x and y.
    params: array of length 10, used as coefficients for the basis functions.
    
    Basis functions (10 total) - expanded to capture more frequencies and interactions:
    0: 1 (constant)
    1: cos(x)
    2: sin(x)
    3: cos(y)
    4: sin(y)
    5: cos(2*x) * cos(2*y)  -- 2nd harmonic interaction
    6: sin(2*x) * sin(2*y)  -- 2nd harmonic interaction
    7: cos(x + y)
    8: sin(x + y)
    9: cos(x - y)
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Precompute trig arguments
    cx = np.cos(x)
    sx = np.sin(x)
    cy = np.cos(y)
    sy = np.sin(y)
    
    # 2nd harmonics
    c2x = np.cos(2 * x)
    s2x = np.sin(2 * x)
    c2y = np.cos(2 * y)
    s2y = np.sin(2 * y)
    
    # Interaction terms
    cxy_plus = np.cos(x + y)
    sxy_plus = np.sin(x + y)
    cxy_minus = np.cos(x - y)
    
    # Combine with parameters
    result = (
        p0
        + p1 * cx
        + p2 * sx
        + p3 * cy
        + p4 * sy
        + p5 * c2x * c2y
        + p6 * s2x * s2y
        + p7 * cxy_plus
        + p8 * sxy_plus
        + p9 * cxy_minus
    )
    
    return result
# EVOLVE-BLOCK-END
