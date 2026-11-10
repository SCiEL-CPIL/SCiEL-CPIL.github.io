# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form expression using a Gaussian Radial Basis Function (RBF) basis.
    This approach uses 10 fixed Gaussian kernels centered evenly across the input domain [-3, 3].
    The parameters are the weights for these kernels.
    This provides a local, non-polynomial basis that can approximate a wide variety of smooth functions.
    """
    # Ensure x is a 1D array
    x = np.asarray(x).flatten()
    
    # Define fixed centers for the Gaussian basis functions
    # 10 centers evenly spaced from -3 to 3
    centers = np.linspace(-3.0, 3.0, 10)
    
    # Define the width (sigma) of the Gaussians
    # A width of 1.5 provides good overlap between adjacent centers (spacing ~0.67)
    sigma = 1.5
    
    # Compute the basis matrix
    # Shape: (n_points, 10)
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    basis = np.exp(-0.5 * (diff / sigma) ** 2)
    
    # Prediction: linear combination of basis functions
    pred = basis @ params
    
    return pred
# EVOLVE-BLOCK-END
