"""
Strategy: Adaptive Gaussian Basis Functions with Learnable Centers and Scales.

This approach differs from the current Chebyshev series by using local basis functions
(Gaussian bumps) whose centers and scales are also learnable parameters. This allows
the model to adapt its resolution where the function has high curvature or sharp features,
rather than using a fixed global polynomial basis.

Params layout (length 10):
- params[0:5]: Centers of the 5 Gaussian bumps
- params[5:10]: Scales (widths) of the 5 Gaussian bumps

The weights of the Gaussians are fixed to 1.0 to keep the parameter count at 10.
The BFGS optimizer will adjust centers and scales to best fit the data.
"""
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Approximates the target function using 5 Gaussian basis functions with learnable
    centers and scales.
    """
    # Extract centers and scales from params
    centers = params[0:5]
    scales = params[5:10]
    
    # Ensure scales are positive to avoid numerical issues
    # Use softplus-like transformation: scale = exp(s) to keep positive
    # But since BFGS starts at 1, we can just use abs + small epsilon
    scales = np.abs(scales) + 1e-6
    
    # Compute Gaussian basis functions
    # x has shape (n,), centers and scales have shape (5,)
    # We want to compute sum_i exp(-(x - c_i)^2 / (2 * s_i^2))
    
    # Reshape for broadcasting: x (n,1), centers (1,5), scales (1,5)
    x_col = x.reshape(-1, 1)
    c_row = centers.reshape(1, -1)
    s_row = scales.reshape(1, -1)
    
    # Compute squared differences
    diff = x_col - c_row  # shape (n, 5)
    
    # Compute Gaussian values
    gaussians = np.exp(-diff ** 2 / (2.0 * s_row ** 2))  # shape (n, 5)
    
    # Sum over the 5 basis functions (weights are 1)
    result = np.sum(gaussians, axis=1)
    
    return result
