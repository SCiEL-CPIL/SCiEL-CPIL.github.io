"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Approximates the target function using a linear trend plus 3 Gaussian Radial Basis Functions.
    This model is flexible enough to capture local features and non-polynomial shapes
    that global polynomials might miss.
    
    y = p[0] * x + 
        p[1] * exp(-((x - p[2])^2) / (2 * p[3]^2)) +
        p[4] * exp(-((x - p[5])^2) / (2 * p[6]^2)) +
        p[7] * exp(-((x - p[8])^2) / (2 * p[9]^2))
    """
    # Unpack parameters
    slope = params[0]
    
    # Gaussian 1
    amp1 = params[1]
    center1 = params[2]
    width1 = params[3]
    
    # Gaussian 2
    amp2 = params[4]
    center2 = params[5]
    width2 = params[6]
    
    # Gaussian 3
    amp3 = params[7]
    center3 = params[8]
    width3 = params[9]
    
    # Calculate Gaussian terms
    # Add small epsilon to width squared to prevent division by zero
    eps = 1e-6
    
    w1_sq = width1 * width1 + eps
    w2_sq = width2 * width2 + eps
    w3_sq = width3 * width3 + eps
    
    g1 = np.exp(-((x - center1) ** 2) / (2.0 * w1_sq))
    g2 = np.exp(-((x - center2) ** 2) / (2.0 * w2_sq))
    g3 = np.exp(-((x - center3) ** 2) / (2.0 * w3_sq))
    
    # Combine
    result = slope * x + amp1 * g1 + amp2 * g2 + amp3 * g3
    
    return result
# EVOLVE-BLOCK-END
