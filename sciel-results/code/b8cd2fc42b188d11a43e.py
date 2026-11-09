"""Symbolic regression with flexible basis and robust fitting."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form prediction using a rich basis of functions.
    params: array of length 10, coefficients for each basis function.
    """
    x = np.asarray(x, dtype=float).ravel()
    n = len(x)
    
    # Build basis matrix with 10 columns
    # 1. constant
    # 2. x
    # 3. x^2
    # 4. x^3
    # 5. x^4
    # 6. sin(x)
    # 7. cos(x)
    # 8. tanh(x)
    # 9. exp(-x^2)  (Gaussian-like)
    # 10. x * sin(x)
    
    basis = np.empty((n, 10), dtype=float)
    basis[:, 0] = 1.0
    basis[:, 1] = x
    basis[:, 2] = x ** 2
    basis[:, 3] = x ** 3
    basis[:, 4] = x ** 4
    basis[:, 5] = np.sin(x)
    basis[:, 6] = np.cos(x)
    basis[:, 7] = np.tanh(x)
    basis[:, 8] = np.exp(-x ** 2)
    basis[:, 9] = x * np.sin(x)
    
    # Predict: dot product of basis with params
    pred = basis @ params
    
    # Ensure finite output
    if not np.all(np.isfinite(pred)):
        pred = np.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
    
    return pred
# EVOLVE-BLOCK-END
