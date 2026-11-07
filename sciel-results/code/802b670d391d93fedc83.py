"""Symbolic regression with a fixed rich basis of 10 functions."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

_START = _time.time()
_DEADLINE = _START + 135.0  # safety margin before 150s kill


def _compute_basis(x: np.ndarray) -> np.ndarray:
    """Compute the 10 basis functions evaluated at x.
    
    Returns shape (n, 10) where columns are:
    0: 1
    1: x
    2: x^2
    3: x^3
    4: x^4
    5: sin(x)
    6: cos(x)
    7: tanh(x)
    8: exp(-x^2)
    9: exp(-|x|)
    """
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    
    basis = np.empty((n, 10), dtype=np.float64)
    basis[:, 0] = 1.0
    basis[:, 1] = x
    basis[:, 2] = x ** 2
    basis[:, 3] = x ** 3
    basis[:, 4] = x ** 4
    basis[:, 5] = np.sin(x)
    basis[:, 6] = np.cos(x)
    basis[:, 7] = np.tanh(x)
    basis[:, 8] = np.exp(-x ** 2)
    basis[:, 9] = np.exp(-np.abs(x))
    
    return basis


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Predict y = sum_i params[i] * basis_i(x) for each x.
    
    Args:
        x: array of input values
        params: array of 10 weights
    
    Returns:
        array of predictions, one per input
    """
    x = np.asarray(x, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)
    
    # Time guard
    if _time.time() > _DEADLINE:
        # Return a simple fallback
        return params[0] * np.ones_like(x) + params[1] * x
    
    basis = _compute_basis(x)
    
    # Linear combination
    pred = basis @ params
    
    # Ensure finite output
    if not np.all(np.isfinite(pred)):
        pred = np.nan_to_num(pred, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return pred
# EVOLVE-BLOCK-END
