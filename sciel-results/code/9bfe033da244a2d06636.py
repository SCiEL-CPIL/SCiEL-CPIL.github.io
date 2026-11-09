"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE_MARGIN = 5.0  # safety margin in seconds before 150s kill

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers the unknown function using localized Gaussian RBFs with fixed centers
    and varying widths, plus a global linear trend. Linear in params for fast BFGS convergence.
    
    Basis functions (10 total):
    0: 1 (intercept)
    1: x (linear trend)
    2-9: Gaussian RBFs exp(-((x-c_i)^2)/(2*w_i^2)) with fixed centers c_i and widths w_i
    """
    x = np.asarray(x, dtype=np.float64)
    original_shape = x.shape
    x_flat = x.flatten()
    
    if x_flat.size == 0:
        return np.array([]).reshape(original_shape)
    
    # Deadline guard
    if time.time() - _START_TIME > 150.0 - _DEADLINE_MARGIN:
        return np.zeros_like(x_flat).reshape(original_shape)
    
    # Fixed centers spread across [-3, 3]
    centers = np.array([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 2.5, 3.0])
    # Varying widths: narrow to wide for local and global features
    widths = np.array([0.5, 0.8, 1.0, 1.2, 1.5, 1.8, 2.0, 2.5])
    
    # Compute basis functions
    b0 = np.ones_like(x_flat)
    b1 = x_flat
    
    # Gaussian RBFs
    rbf_terms = np.zeros((8, x_flat.size))
    for i in range(8):
        diff = x_flat - centers[i]
        rbf_terms[i] = np.exp(-(diff * diff) / (2.0 * widths[i] * widths[i]))
    
    # Linear combination
    y = params[0] * b0 + params[1] * b1
    for i in range(8):
        y += params[2 + i] * rbf_terms[i]
    
    # Ensure finite values
    y = np.where(np.isfinite(y), y, 0.0)
    
    return y.reshape(original_shape)
# EVOLVE-BLOCK-END
