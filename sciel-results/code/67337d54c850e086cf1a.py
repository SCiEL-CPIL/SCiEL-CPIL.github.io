# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Structural change: use a complete quadratic basis in normalized inputs.
    # This captures all pairwise interactions and squared terms, giving a richer
    # representation for smooth non-linear targets than the linear seed.
    # The model is linear in params, so BFGS converges reliably.
    #
    # Normalization: map inputs from [0.5, 2] to [0, 1] for numerical stability.
    #   xn = (x - 0.5) / 1.5
    #   yn = (y - 0.5) / 1.5
    #   zn = (z - 0.5) / 1.5
    #
    # Complete quadratic basis in 3 variables (10 terms):
    #   0: 1 (bias)
    #   1: xn
    #   2: yn
    #   3: zn
    #   4: xn^2
    #   5: yn^2
    #   6: zn^2
    #   7: xn*yn
    #   8: xn*zn
    #   9: yn*zn
    
    # Guard against non-finite inputs
    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y)) and np.all(np.isfinite(z))):
        n = len(x)
        return np.zeros(n, dtype=np.float64)
    
    n = len(x)
    
    # Normalize inputs to [0, 1]
    xn = (x - 0.5) / 1.5
    yn = (y - 0.5) / 1.5
    zn = (z - 0.5) / 1.5
    
    # Clip to avoid any floating point edge cases
    xn = np.clip(xn, 0.0, 1.0)
    yn = np.clip(yn, 0.0, 1.0)
    zn = np.clip(zn, 0.0, 1.0)
    
    # Build feature matrix: shape (n, 10)
    F = np.empty((n, 10), dtype=np.float64)
    F[:, 0] = 1.0
    F[:, 1] = xn
    F[:, 2] = yn
    F[:, 3] = zn
    F[:, 4] = xn * xn
    F[:, 5] = yn * yn
    F[:, 6] = zn * zn
    F[:, 7] = xn * yn
    F[:, 8] = xn * zn
    F[:, 9] = yn * zn
    
    # Compute prediction
    pred = F @ params
    
    # Ensure finite output
    if not np.all(np.isfinite(pred)):
        pred = np.nan_to_num(pred, nan=0.0, posinf=1e10, neginf=-1e10)
    
    return pred
# EVOLVE-BLOCK-END
