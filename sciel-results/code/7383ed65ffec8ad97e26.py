"""Symbolic regression with 1 input; Gaussian Process-style RBF kernel formulation."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_TIME_LIMIT = 145.0  # Safety margin before 150s kill


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Gaussian Process Regression (Kernel) formulation.
    
    The prediction is a linear combination of 10 squared-exponential (RBF) kernels
    centered at fixed, evenly spaced points across the domain [0.1, 6].
    
    The 10 parameters represent the kernel weights for each center.
    
    This shifts from polynomial basis approximation to a local interpolation/smoothing
    approach that can adapt to arbitrary non-polynomial smooth functions.
    
    The evenly-spaced centers with moderate sigma provide good coverage of the domain
    while maintaining numerical stability for BFGS optimization.
    """
    x = np.asarray(x, dtype=float).flatten()
    params = np.asarray(params, dtype=float).flatten()
    
    # Ensure exactly 10 parameters
    if len(params) < 10:
        params = np.concatenate([params, np.zeros(10 - len(params))])
    else:
        params = params[:10]
    
    # Define 10 evenly-spaced centers across [0.1, 6]
    x_min, x_max = 0.1, 6.0
    n_centers = 10
    centers = np.linspace(x_min + (x_max - x_min) / (2 * n_centers),
                          x_max - (x_max - x_min) / (2 * n_centers), n_centers)
    
    # Use a moderate sigma for smooth interpolation
    # sigma = 0.8 provides good balance between local adaptivity and smoothness
    sigma = 0.8
    
    # Compute RBF basis: shape (N, n_centers)
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    basis = np.exp(-0.5 * (diff / sigma) ** 2)  # shape (N, 10)
    
    # Linear combination of kernel weights
    y = basis @ params
    
    return y
# EVOLVE-BLOCK-END
