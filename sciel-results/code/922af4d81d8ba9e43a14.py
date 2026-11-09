"""Rewritten symbolic regression candidate: localized RBF + polynomial hybrid basis."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form expression using a fixed set of 10 basis functions.
    Combines polynomial terms, trigonometric terms, and localized Gaussian RBFs.
    params: length-10 array of coefficients (fitted by BFGS).
    Returns an array with one prediction per point.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    
    # Ensure x, y are 1D arrays
    x_flat = x.ravel()
    y_flat = y.ravel()
    
    # Basis functions:
    # 0: constant
    # 1: x
    # 2: y
    # 3: x^2
    # 4: y^2
    # 5: x*y
    # 6: sin(x)
    # 7: cos(y)
    # 8: Gaussian RBF centered at (0,0) with sigma=0.5
    # 9: Gaussian RBF centered at (0.5,0.5) with sigma=0.4
    
    b0 = np.ones_like(x_flat)
    b1 = x_flat
    b2 = y_flat
    b3 = x_flat ** 2
    b4 = y_flat ** 2
    b5 = x_flat * y_flat
    b6 = np.sin(x_flat)
    b7 = np.cos(y_flat)
    
    # RBF 8: center (0,0), sigma=0.5
    r8 = np.sqrt(x_flat ** 2 + y_flat ** 2)
    b8 = np.exp(-r8 ** 2 / (2 * 0.5 ** 2))
    
    # RBF 9: center (0.5, 0.5), sigma=0.4
    r9 = np.sqrt((x_flat - 0.5) ** 2 + (y_flat - 0.5) ** 2)
    b9 = np.exp(-r9 ** 2 / (2 * 0.4 ** 2))
    
    # Combine basis functions with parameters
    p = params
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
