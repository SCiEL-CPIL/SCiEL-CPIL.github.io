"""Symbolic regression with 1 input using a hybrid Chebyshev + Gaussian kernel basis."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Fits a hybrid model combining global Chebyshev trends and local Gaussian kernels.
    
    The 10 parameters are allocated as:
    - params[0:4]: Weights for Chebyshev polynomials T_0, T_1, T_2, T_3.
    - params[4:10]: Weights for 6 Gaussian kernels centered at evenly spaced points in the domain.
    
    This structure allows the model to capture both the overall shape of the function
    and local deviations, providing a more flexible approximation than a pure 
    polynomial of the same degree, while maintaining numerical stability for BFGS.
    """
    x = np.asarray(x, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)
    
    if len(x) == 0:
        return np.empty(0, dtype=np.float64)
    
    # Domain is [0.1, 6.0]
    x_min = 0.1
    x_max = 6.0
    center = (x_min + x_max) / 2.0
    scale = (x_max - x_min) / 2.0
    
    # Clip to valid domain to prevent extrapolation issues during optimization
    x_safe = np.clip(x, x_min, x_max)
    
    # Map to normalized coordinate u in [-1, 1] for stable Chebyshev evaluation
    u = (x_safe - center) / scale
    
    n = len(x)
    
    # Initialize basis matrix
    # Columns 0-3: Chebyshev
    # Columns 4-9: Gaussian Kernels
    basis = np.empty((n, 10), dtype=np.float64)
    
    # --- Global Component: Chebyshev Polynomials ---
    # T_0(u) = 1
    basis[:, 0] = 1.0
    # T_1(u) = u
    basis[:, 1] = u
    # T_2(u) = 2u^2 - 1
    basis[:, 2] = 2.0 * u * u - 1.0
    # T_3(u) = 4u^3 - 3u
    basis[:, 3] = 4.0 * u * u * u - 3.0 * u
    
    # --- Local Component: Gaussian Kernels ---
    # We use 6 kernels to utilize the remaining 6 parameters.
    # Centers are evenly spaced in the original x-domain [0.1, 6.0]
    # Mapping centers to u-domain:
    # x_centers = np.linspace(0.1, 6.0, 6)
    # u_centers = (x_centers - center) / scale
    # This is equivalent to np.linspace(-1, 1, 6)
    kernel_centers = np.linspace(-1.0, 1.0, 6)
    
    # Bandwidth (sigma) in u-units.
    # A sigma of 0.3 provides good overlap for smoothness but enough localization for adaptation.
    sigma = 0.3
    sigma_sq_2 = 2.0 * sigma * sigma
    
    for i, c in enumerate(kernel_centers):
        # Gaussian kernel: exp(-((u - c)^2) / (2 * sigma^2))
        diff = u - c
        basis[:, 4 + i] = np.exp(-(diff * diff) / sigma_sq_2)
    
    # Linear combination
    return basis @ params

# EVOLVE-BLOCK-END
