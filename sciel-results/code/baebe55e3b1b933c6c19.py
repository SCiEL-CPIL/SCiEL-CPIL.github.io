"""Symbolic regression with 1 input using Gaussian Process kernel basis."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Predict using a sum of squared-exponential (Gaussian) kernels centered at
    fixed evenly-spaced points across the domain [0.1, 6.0].
    
    params: length-10 array.
      - params[0:9] are the weights for 9 kernel centers.
      - params[9] is a constant offset (intercept).
    
    The kernel width is fixed at a reasonable value for the domain.
    """
    x = np.asarray(x, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)
    
    if len(x) == 0:
        return np.empty(0, dtype=np.float64)
    
    # 9 kernel centers evenly spaced across [0.1, 6.0]
    centers = np.linspace(0.1, 6.0, 9)
    
    # Fixed kernel width (bandwidth) for the squared-exponential kernel
    # Domain length is ~5.9, so a bandwidth of ~1.0 gives reasonable overlap
    bandwidth = 1.0
    
    # Compute kernel values: K(x, c_i) = exp(-0.5 * ((x - c_i) / bandwidth)^2)
    # Shape: (n_points, 9)
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    kernel_vals = np.exp(-0.5 * (diff / bandwidth) ** 2)
    
    # Linear combination of kernel values
    weights = params[0:9]
    offset = params[9]
    
    prediction = kernel_vals @ weights + offset
    
    return prediction
# EVOLVE-BLOCK-END
