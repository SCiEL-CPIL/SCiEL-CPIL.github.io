# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    p = params
    
    # Use 10 localized Gaussian basis functions centered at evenly spaced points
    # across [-3, 3] with fixed widths for local flexibility
    # Centers: -3, -2.4, -1.8, -1.2, -0.6, 0, 0.6, 1.2, 1.8, 2.4
    # Actually, let's use centers that span the full range nicely
    centers = np.linspace(-3.0, 3.0, 10)
    width = 0.8  # Fixed width for all Gaussians
    
    # Compute all Gaussian basis functions: exp(-(x - center)^2 / (2 * width^2))
    # x shape: (N,), centers shape: (10,)
    # We need output shape (N,)
    
    # Efficient computation using broadcasting
    # diff[i, j] = x[i] - centers[j]
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    gaussians = np.exp(-diff**2 / (2.0 * width**2))
    
    # out[i] = sum_j p[j] * gaussians[i, j]
    out = gaussians @ p
    
    return out
# EVOLVE-BLOCK-END
