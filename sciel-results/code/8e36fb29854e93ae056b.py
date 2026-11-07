# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    RBF network: 5 centers, 5 learnable weights, fixed widths.
    params[0:5] = centers c_i
    params[5:10] = weights w_i
    Fixed sigmas chosen to cover a range of scales.
    """
    centers = params[0:5]
    weights = params[5:10]
    
    # Fixed widths at multiple scales
    sigmas = np.array([0.25, 0.5, 1.0, 2.0, 4.0])
    
    # Compute differences: (N, 5)
    diffs = x[:, np.newaxis] - centers[np.newaxis, :]
    
    # Gaussian kernels
    kernels = np.exp(-(diffs ** 2) / (sigmas[np.newaxis, :] ** 2))
    
    # Weighted sum
    y = np.sum(kernels * weights[np.newaxis, :], axis=1)
    
    return y
# EVOLVE-BLOCK-END
