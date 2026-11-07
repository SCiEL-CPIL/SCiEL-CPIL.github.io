"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression using a fixed basis of 10 features.
    The 10 params weight each basis function.
    Basis: [1, x, y, x^2, y^2, x*y, x^3, y^3, sin(x), sin(y)]
    """
    x = np.asarray(x, dtype=np.float64).flatten()
    y = np.asarray(y, dtype=np.float64).flatten()
    p = np.asarray(params, dtype=np.float64).flatten()
    
    # Ensure we have exactly 10 params; pad or truncate if needed
    if p.shape[0] < 10:
        p = np.concatenate([p, np.ones(10 - p.shape[0])])
    else:
        p = p[:10]
    
    # Build basis matrix: each column is one basis function evaluated at all points
    # 10 basis functions
    b0 = np.ones_like(x)          # constant
    b1 = x                        # x
    b2 = y                        # y
    b3 = x * x                    # x^2
    b4 = y * y                    # y^2
    b5 = x * y                    # x*y
    b6 = x * x * x                # x^3
    b7 = y * y * y                # y^3
    b8 = np.sin(x)                # sin(x)
    b9 = np.sin(y)                # sin(y)
    
    # Prediction is weighted sum of basis functions
    pred = (p[0] * b0 + p[1] * b1 + p[2] * b2 + p[3] * b3 + 
            p[4] * b4 + p[5] * b5 + p[6] * b6 + p[7] * b7 + 
            p[8] * b8 + p[9] * b9)
    
    # Ensure finite output
    if not np.all(np.isfinite(pred)):
        pred = np.where(np.isfinite(pred), pred, 0.0)
    
    return pred
# EVOLVE-BLOCK-END
