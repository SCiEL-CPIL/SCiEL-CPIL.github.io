import numpy as np
import time

# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid basis: first 5 params are Chebyshev polynomial coefficients (global smooth fit),
    last 5 params are Gaussian RBF amplitudes with fixed centers (local flexibility).
    
    This allows the model to capture both smooth global trends and local features
    like sharp transitions or discontinuities.
    """
    original_shape = x.shape
    x_flat = x.flatten()
    N = len(x_flat)
    
    if N == 0:
        return np.array([], dtype=float)
    
    p = params.flatten()
    
    # Part 1: Chebyshev polynomial basis (first 5 params)
    # Scale x to [-1, 1] for Chebyshev polynomials
    t = x_flat / 2.0
    
    # Compute Chebyshev polynomials T_0(t) to T_4(t)
    B_cheb = np.empty((N, 5), dtype=float)
    B_cheb[:, 0] = 1.0
    B_cheb[:, 1] = t
    for k in range(2, 5):
        B_cheb[:, k] = 2.0 * t * B_cheb[:, k-1] - B_cheb[:, k-2]
    
    # Part 2: Gaussian RBF basis (last 5 params)
    # Fixed centers spread across [-2, 2]
    centers = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    width = 0.5  # Fixed width for Gaussian kernels
    
    # Compute RBF basis: exp(-((x - center)^2) / (2 * width^2))
    # Shape: (N, 5)
    diff = x_flat[:, np.newaxis] - centers[np.newaxis, :]
    B_rbf = np.exp(-(diff ** 2) / (2.0 * width ** 2))
    
    # Combine: y = B_cheb @ p[:5] + B_rbf @ p[5:]
    y = B_cheb @ p[:5] + B_rbf @ p[5:]
    
    return y.reshape(original_shape)
# EVOLVE-BLOCK-END
