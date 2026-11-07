"""Symbolic regression with RBF kernel basis."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Model: linear combination of Gaussian RBF kernels with fixed centers.
    10 parameters = 10 RBF weights.
    Centers are evenly spaced in [-3, 3].
    Widths are fixed at a value that gives good local interpolation.
    
    params[0:10] -> weights for each RBF kernel
    """
    # Fixed centers evenly spaced in [-3, 3]
    centers = np.linspace(-3.0, 3.0, 10)
    # Fixed bandwidth: chosen to balance local flexibility and smoothness
    bandwidth = 1.2
    
    # Compute RBF values: (n_points, 10)
    # x shape: (n,), centers shape: (10,)
    diff = x[:, np.newaxis] - centers[np.newaxis, :]  # (n, 10)
    rbf = np.exp(-0.5 * (diff / bandwidth) ** 2)  # (n, 10)
    
    # Linear combination
    return rbf @ params
# EVOLVE-BLOCK-END
