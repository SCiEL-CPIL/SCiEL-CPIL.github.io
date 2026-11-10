"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    RBF network with 10 fixed Gaussian centers spanning [0.1, 6].
    params[0:10]: weights for each of the 10 RBF centers.
    
    The centers are fixed at equally spaced points in the domain.
    The width (sigma) is fixed at 1.0 to ensure a convex optimization landscape
    for the weights, avoiding local minima associated with learning the width.
    """
    # Fixed centers spanning the input domain [0.1, 6]
    # Using 10 centers to utilize all 10 weight parameters
    centers = np.linspace(0.1, 6.0, 10)
    
    # Fixed width for stable, convex optimization of weights
    sigma = 1.0
    
    # Compute distances from x to each center
    # x shape: (N,)
    # centers shape: (10,)
    # diff shape: (N, 10)
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    
    # Compute Gaussian kernels
    # exp(-diff^2 / (2 * sigma^2))
    kernel = np.exp(-diff**2 / (2.0 * sigma**2))
    
    # Linear combination of kernels
    # weights shape: (10,)
    weights = params[0:10]
    
    # prediction = kernel @ weights
    prediction = kernel @ weights
    
    return prediction
# EVOLVE-BLOCK-END
