"""Symbolic regression with 1 input; Gaussian Process (RBF kernel) formulation."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_TIME_LIMIT = 140.0  # Safety margin before 150s kill


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Gaussian Process Regression formulation using a sum of squared-exponential
    (RBF) kernels centered at fixed, evenly spaced points across the domain.
    
    The 10 parameters represent the kernel weights for each center.
    Using exp(params) ensures positive weights, which stabilizes the optimization
    and allows the model to adapt to arbitrary smooth functions via local interpolation.
    
    The input x is scaled to [-1, 1] for numerical stability.
    """
    x = np.asarray(x, dtype=float)
    params = np.asarray(params, dtype=float).flatten()
    
    # Ensure exactly 10 parameters
    if len(params) < 10:
        params = np.concatenate([params, np.ones(10 - len(params))])
    else:
        params = params[:10]
    
    # Scale input x from [0.1, 6] to roughly [-1, 1]
    x_min, x_max = 0.1, 6.0
    x_scaled = (x - (x_min + x_max) / 2.0) / ((x_max - x_min) / 2.0)
    
    # Fixed centers evenly spaced across the scaled domain [-1, 1]
    n_centers = 10
    centers = np.linspace(-1.0, 1.0, n_centers)
    
    # Compute RBF basis: shape (N, n_centers)
    diff = x_scaled[:, np.newaxis] - centers[np.newaxis, :]
    sigma = 0.5  # Fixed bandwidth
    rbf_basis = np.exp(-0.5 * (diff / sigma) ** 2)  # shape (N, n_centers)
    
    # Use exp(params) to ensure positive weights
    weights = np.exp(params)
    
    # Prediction is a weighted sum of RBF kernels
    y = rbf_basis @ weights
    
    return y
# EVOLVE-BLOCK-END
