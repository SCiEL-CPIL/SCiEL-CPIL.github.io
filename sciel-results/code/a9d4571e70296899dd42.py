"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Gaussian RBF network with 5 kernels.
    params[0:5] = centers of the Gaussian kernels
    params[5:10] = amplitudes of the Gaussian kernels
    Fixed width sigma = 0.7 for stability.
    """
    # Ensure params are finite
    if not np.all(np.isfinite(params)):
        params = np.ones(10)
    
    centers = params[0:5]
    amplitudes = params[5:10]
    
    # Fixed width for stability
    sigma = 0.7
    
    # Compute RBF basis: shape (5, N)
    N = x.shape[0]
    if N == 0:
        return np.array([])
    
    # x is shape (N,), centers is shape (5,)
    # diff[i, j] = x[j] - centers[i]
    diff = x[None, :] - centers[:, None]  # shape (5, N)
    
    # RBF values: exp(-diff^2 / (2 * sigma^2))
    rbf_values = np.exp(-diff ** 2 / (2.0 * sigma ** 2))  # shape (5, N)
    
    # Output: sum of amplitudes * rbf_values
    # amplitudes shape (5,), rbf_values shape (5, N)
    result = amplitudes @ rbf_values  # shape (N,)
    
    return result
# EVOLVE-BLOCK-END
