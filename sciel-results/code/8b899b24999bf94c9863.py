"""Hybrid basis: Polynomials, Gaussian, and Trigonometric terms."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers the target function using a hybrid basis of polynomials,
    a Gaussian bump, and trigonometric functions.
    
    Basis functions:
    0: 1
    1: x
    2: y
    3: x^2
    4: y^2
    5: x*y
    6: exp(-(x^2 + y^2))
    7: sin(x)
    8: sin(y)
    9: cos(x + y)
    """
    p = params
    
    # Polynomial terms
    b0 = np.ones_like(x)
    b1 = x
    b2 = y
    b3 = x * x
    b4 = y * y
    b5 = x * y
    
    # Non-polynomial terms
    r2 = x * x + y * y
    b6 = np.exp(-r2)
    
    b7 = np.sin(x)
    b8 = np.sin(y)
    b9 = np.cos(x + y)
    
    # Linear combination
    result = (
        p[0] * b0 +
        p[1] * b1 +
        p[2] * b2 +
        p[3] * b3 +
        p[4] * b4 +
        p[5] * b5 +
        p[6] * b6 +
        p[7] * b7 +
        p[8] * b8 +
        p[9] * b9
    )
    
    return result
# EVOLVE-BLOCK-END
