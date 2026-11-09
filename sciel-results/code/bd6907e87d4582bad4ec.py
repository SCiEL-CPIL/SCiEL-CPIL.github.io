"""Symbolic regression with 2 inputs: 10-parameter basis mixing low-degree polynomials and
trigonometric functions to capture both polynomial and periodic structure."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_T0 = time.time()
_DEADLINE = 145.0


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    10-term basis for f(x, y) with x, y in [-1, 1].
    
    Basis terms:
    0: 1
    1: x
    2: y
    3: x^2
    4: y^2
    5: x*y
    6: sin(pi*x)
    7: sin(pi*y)
    8: cos(pi*x)
    9: cos(pi*y)
    
    This mixes polynomial and trigonometric features to capture both smooth
    polynomial trends and periodic/oscillatory behavior.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    
    if time.time() > _DEADLINE:
        # Fallback: simple mean prediction
        return np.mean(y) * np.ones_like(x)
    
    # Compute the 10 basis features
    f0 = np.ones_like(x)
    f1 = x
    f2 = y
    f3 = x * x
    f4 = y * y
    f5 = x * y
    f6 = np.sin(np.pi * x)
    f7 = np.sin(np.pi * y)
    f8 = np.cos(np.pi * x)
    f9 = np.cos(np.pi * y)
    
    F = np.column_stack([f0, f1, f2, f3, f4, f5, f6, f7, f8, f9])
    
    pred = F @ params
    
    # Ensure finite output
    if not np.all(np.isfinite(pred)):
        pred = np.where(np.isfinite(pred), pred, 0.0)
    
    return pred
# EVOLVE-BLOCK-END
