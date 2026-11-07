"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers the target function using a Gaussian Radial Basis Function (RBF) expansion.
    
    The model uses 10 Gaussian RBFs with fixed centers and widths.
    The 10 parameters correspond to the weights of these basis functions.
    
    RBFs are well-suited for approximating unknown smooth functions, especially
    if they contain localized features or non-polynomial behavior, which might
    cause low-degree polynomials to oscillate or fit poorly.
    """
    # Define fixed hyperparameters for the RBF basis
    # 10 centers evenly distributed across the input domain [-3, 3]
    centers = np.linspace(-3.0, 3.0, 10)
    
    # Width of the Gaussians. 
    # A width of 1.0 provides good overlap for a domain of length 6 with 10 points.
    sigma = 1.0
    
    # Ensure x is a 1D array for broadcasting
    x = np.asarray(x).flatten()
    
    # Compute the RBF basis matrix
    # shape: (num_points, 10)
    # B[i, j] = exp(-(x_i - centers[j])^2 / (2 * sigma^2))
    
    # Calculate squared distances
    # x[:, None] has shape (N, 1)
    # centers[None, :] has shape (1, 10)
    diff = x[:, None] - centers[None, :]
    
    # Compute the exponential term
    # Squared term: diff^2
    # Divide by 2*sigma^2
    # Exponentiate
    basis = np.exp(-diff**2 / (2 * sigma**2))
    
    # The model is a linear combination of the basis functions
    # result = basis @ params
    # params has shape (10,)
    # basis has shape (N, 10)
    # result has shape (N,)
    
    result = basis @ params
    
    return result
# EVOLVE-BLOCK-END
