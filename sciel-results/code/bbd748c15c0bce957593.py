"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    RBF network with 9 fixed Gaussian centers spanning [0.1, 6].
    params[0]: learnable width (sigma) of the Gaussian kernels.
    params[1:10]: weights for each of the 9 RBF centers.
    
    The centers are fixed at equally spaced points in the domain.
    Sigma is parameterized as exp(params[0] - 1.0) to start near 1.0.
    """
    # Fixed centers spanning the input domain [0.1, 6]
    # Using 9 centers to utilize all 9 weight parameters
    centers = np.linspace(0.1, 6.0, 9)
    
    # Extract parameters
    # sigma = exp(params[0] - 1.0)
    # If params[0] = 1, sigma = 1.0.
    # If params[0] = 0, sigma = e^-1 ~ 0.37.
    # If params[0] = 2, sigma = e^1 ~ 2.7.
    # This provides a good range of widths.
    sigma = np.exp(params[0] - 1.0)
    
    # Clamp sigma to prevent numerical issues if optimization goes extreme
    sigma = np.clip(sigma, 1e-3, 10.0)
    
    # Compute distances from x to each center
    # x shape: (N,)
    # centers shape: (9,)
    # diff shape: (N, 9)
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    
    # Compute Gaussian kernels
    # exp(-diff^2 / (2 * sigma^2))
    kernel = np.exp(-diff**2 / (2.0 * sigma**2))
    
    # Linear combination of kernels
    # weights shape: (9,)
    weights = params[1:10]
    
    # prediction = kernel @ weights
    prediction = kernel @ weights
    
    return prediction
# EVOLVE-BLOCK-END
