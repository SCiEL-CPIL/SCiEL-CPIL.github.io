"""Rational function with quadratic numerator and trigonometric-bounded denominator."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Rational function: quadratic numerator / trigonometric-bounded denominator.
    - params[0..5]: numerator coefficients for [1, x, y, x^2, y^2, xy]
    - params[6..9]: denominator modulation: 1 + p[6]*cos(x) + p[7]*cos(y) + p[8]*cos(x+y) + p[9]*cos(x-y)
    Denominator is clamped to avoid zero-crossings.
    """
    p = params
    
    # Numerator: full quadratic in x, y
    num = (
        p[0]
        + p[1] * x
        + p[2] * y
        + p[3] * x * x
        + p[4] * y * y
        + p[5] * x * y
    )
    
    # Denominator: 1 + trigonometric terms (bounded, always > 0 after clamp)
    denom = (
        1.0
        + p[6] * np.cos(x)
        + p[7] * np.cos(y)
        + p[8] * np.cos(x + y)
        + p[9] * np.cos(x - y)
    )
    
    # Clamp denominator to avoid division by zero or sign flips
    denom = np.maximum(denom, 0.1)
    denom = np.minimum(denom, 10.0)
    
    pred = num / denom
    
    # Ensure finite output
    pred = np.where(np.isfinite(pred), pred, 0.0)
    
    return pred
# EVOLVE-BLOCK-END
