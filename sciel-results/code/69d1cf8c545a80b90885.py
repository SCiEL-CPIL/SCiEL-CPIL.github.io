"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    RBF network with 8 fixed Gaussian centers spanning [0.1, 6].
    params[0]: log-width parameter (sigma = exp(params[0])).
    params[1:9]: weights for each of the 8 RBF centers.
    
    The centers are fixed at equally spaced points in the domain.
    Using exp(params[0]) ensures sigma > 0 and is differentiable.
    """
    # Fixed centers spanning the input domain [0.1, 6]
    centers = np.linspace(0.1, 6.0, 8)
    
    # Extract parameters
    # Ensure sigma is positive and reasonable
    # params[0] starts at 1.0, so sigma starts at e^1 ≈ 2.718
    # If params[0] is very negative, sigma becomes very small (sharp peaks)
    # If params[0] is very positive, sigma becomes very large (broad peaks)
    log_sigma = params[0]
    # Clamp log_sigma to avoid numerical issues
    log_sigma = np.clip(log_sigma, -5, 5)
    sigma = np.exp(log_sigma)
    
    # Compute Gaussian RBF values
    # x has shape (n,), centers has shape (8,)
    # diff has shape (n, 8)
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    rbf_values = np.exp(-0.5 * (diff / sigma) ** 2)
    
    # Weighted sum
    weights = params[1:9]
    prediction = rbf_values @ weights
    
    return prediction
# EVOLVE-BLOCK-END
