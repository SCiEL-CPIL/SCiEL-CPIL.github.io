"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers an unknown function using a Gaussian Radial Basis Function (RBF) network.
    This approach uses 10 Gaussian kernels centered at fixed locations in [-3, 3].
    The parameters correspond to the weights of these kernels.
    
    y(x) = sum_{i=0}^{9} params[i] * exp(-((x - centers[i])^2) / (2 * sigma^2))
    
    This model is differentiable, smooth, and can approximate a wide variety of functions.
    """
    # Ensure x is a numpy array
    x = np.asarray(x, dtype=np.float64)
    
    # Number of basis functions (parameters)
    n_basis = 10
    
    # Fixed centers for the Gaussian kernels, uniformly spaced in [-3, 3]
    centers = np.linspace(-3.0, 3.0, n_basis)
    
    # Fixed width parameter (sigma) for the Gaussian kernels
    # A smaller sigma makes the basis functions more localized.
    # With 10 centers over a range of 6, spacing is 0.666.
    # Sigma of 0.5 provides good overlap for smooth approximation.
    sigma = 0.5
    sigma_sq = sigma * sigma
    two_sigma_sq = 2.0 * sigma_sq
    
    # Compute prediction: y = sum(params[i] * kernel_i(x))
    # To avoid memory issues for large x, we compute in chunks if necessary, 
    # but for 200 points and 10 kernels, a direct matrix operation is efficient.
    
    # x shape: (N,)
    # centers shape: (10,)
    # We want to compute (N, 10) matrix of kernel values.
    
    # Reshape x for broadcasting: (N, 1)
    x_col = x[:, np.newaxis]
    # Reshape centers for broadcasting: (1, 10)
    c_row = centers[np.newaxis, :]
    
    # Compute squared distances: (N, 10)
    dist_sq = (x_col - c_row) ** 2
    
    # Compute Gaussian values: (N, 10)
    # Using np.exp for element-wise operation
    # We multiply by params to get the final sum
    # y = K @ params
    # K shape: (N, 10), params shape: (10,)
    # Result shape: (N,)
    
    kernel_vals = np.exp(-dist_sq / two_sigma_sq)
    
    # Linear combination of kernel values
    y = kernel_vals.dot(params)
    
    return y
# EVOLVE-BLOCK-END
