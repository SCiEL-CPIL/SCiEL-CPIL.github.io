"""
Symbolic regression with 1 input.
Uses Gaussian RBF kernels with fixed centers for local flexibility.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Fixed centers evenly spaced across [-3, 3]
# 10 centers, spacing = 6/9 = 2/3
_CENTERS = np.linspace(-3.0, 3.0, 10)
# RBF width: chosen to provide local coverage without being too narrow
# With spacing ~0.67, a width of ~0.5 gives good overlap
_RBF_WIDTH = 0.5
_RBF_WIDTH_SQ = _RBF_WIDTH ** 2


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Evaluates a linear combination of Gaussian RBF kernels.
    
    The model is:
        f(x) = sum_{i=0}^{9} params[i] * exp(-(x - c_i)^2 / (2 * sigma^2))
    
    where c_i are fixed centers and sigma is the RBF width.
    The 10 parameters act as adaptive weights for local interpolation.
    """
    # Ensure x is a numpy array
    x = np.asarray(x, dtype=np.float64)
    
    # Compute RBF values: shape (N, 10)
    # diff: (N, 10) - difference between each x and each center
    diff = x[:, np.newaxis] - _CENTERS[np.newaxis, :]
    
    # RBF: exp(-diff^2 / (2 * sigma^2))
    rbf = np.exp(-(diff ** 2) / (2.0 * _RBF_WIDTH_SQ))
    
    # Linear combination of basis functions
    # rbf shape: (N, 10)
    # params shape: (10,)
    # result shape: (N,)
    y = rbf @ params
    
    return y
# EVOLVE-BLOCK-END
