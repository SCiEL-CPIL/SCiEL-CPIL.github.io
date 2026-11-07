"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Approximate target function using a Gaussian RBF network.
    First 5 params define centers, last 5 define amplitudes.
    Uses a fixed width for numerical stability.
    """
    # Sanitize params to avoid non-finite values during optimization
    if not np.all(np.isfinite(params)):
        params = np.ones(10)
    
    # Extract centers and amplitudes
    centers = params[:5]
    amplitudes = params[5:]
    
    # Fixed width for Gaussian kernels
    width = 0.5
    
    # Compute RBF outputs
    # For each center c_i, compute exp(-((x - c_i)^2) / (2 * width^2))
    # Then take weighted sum with amplitudes
    
    # Vectorized computation
    # x shape: (N,)
    # centers shape: (5,)
    # We want a matrix of shape (N, 5) where element [i, j] is RBF_j(x_i)
    
    # diff shape: (N, 5)
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    
    # Compute squared distances
    diff_sq = diff * diff
    
    # Compute Gaussian kernels
    # exp(-diff_sq / (2 * width^2))
    kernel_arg = -diff_sq / (2.0 * width * width)
    
    # Clip to avoid underflow issues (though exp handles it, clipping helps with very large negative values)
    # Actually, np.exp handles -inf well, but let's be safe
    rbf_values = np.exp(kernel_arg)
    
    # Weighted sum: sum over centers (axis=1)
    # rbf_values shape: (N, 5)
    # amplitudes shape: (5,)
    result = np.sum(rbf_values * amplitudes[np.newaxis, :], axis=1)
    
    return result
# EVOLVE-BLOCK-END
