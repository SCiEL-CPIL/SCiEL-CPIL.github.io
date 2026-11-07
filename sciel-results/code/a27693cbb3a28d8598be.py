"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Design: 4 Gaussian kernels + linear trend + constant offset
    # params[0:4] : offsets for 4 kernel centers (base at 1.0, 2.5, 4.0, 5.5)
    # params[4:8] : weights for the 4 kernels (can be negative)
    # params[8]   : linear coefficient
    # params[9]   : constant offset
    
    # Base centers spread across [0.1, 6.0]
    base_centers = np.array([1.0, 2.5, 4.0, 5.5])
    
    # Apply offsets to centers
    centers = base_centers + params[0:4]
    
    # Weights for kernels
    weights = params[4:8]
    
    # Linear coefficient and constant
    linear_coeff = params[8]
    const = params[9]
    
    # Fixed bandwidth - moderate overlap between kernels
    sigma = 1.2
    
    # x shape: (n,)
    # centers shape: (4,)
    # Compute distances: (n, 4)
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    
    # Squared exponential kernel: exp(-diff^2 / (2 * sigma^2))
    kernel = np.exp(-diff ** 2 / (2.0 * sigma ** 2))
    
    # Prediction: weighted sum of kernels + linear term + constant
    # (n, 4) @ (4,) -> (n,)
    pred = kernel @ weights + linear_coeff * x + const
    
    return pred
# EVOLVE-BLOCK-END
