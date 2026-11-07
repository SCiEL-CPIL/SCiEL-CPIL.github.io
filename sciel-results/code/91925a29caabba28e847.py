"""Rewritten program: Hybrid basis with fixed and learnable RBF centers."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid model combining a low-order polynomial basis with Gaussian RBF kernels.
    
    Parameters (length 10):
    - params[0]: overall scale (multiplies everything)
    - params[1:4]: coefficients for polynomial basis [1, x, x^2]
    - params[4:7]: centers for 3 Gaussian RBF kernels
    - params[7:10]: amplitudes for the 3 Gaussian RBF kernels
    
    The polynomial part captures global trends, while the RBF part captures
    local features. Fixed kernel width is used for stability.
    """
    # Ensure x is 1D array
    x = np.asarray(x, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)
    
    # Extract parameters
    scale = params[0]
    poly_coeffs = params[1:4]  # [c0, c1, c2]
    centers = params[4:7]      # 3 centers
    amplitudes = params[7:10]  # 3 amplitudes
    
    # Polynomial basis: [1, x, x^2]
    # Shape: (n_points, 3)
    poly_basis = np.column_stack([
        np.ones_like(x),
        x,
        x ** 2
    ])
    
    # Polynomial contribution
    poly_part = poly_basis @ poly_coeffs  # shape: (n_points,)
    
    # RBF part: Gaussian kernels with fixed width
    width = 0.6
    inv_2w2 = 1.0 / (2.0 * width * width)
    
    # diff shape: (n_points, 3)
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    exponents = -diff * diff * inv_2w2
    rbf_basis = np.exp(exponents)  # shape: (n_points, 3)
    
    # RBF contribution
    rbf_part = rbf_basis @ amplitudes  # shape: (n_points,)
    
    # Combine and scale
    result = scale * (poly_part + rbf_part)
    
    return result
# EVOLVE-BLOCK-END
