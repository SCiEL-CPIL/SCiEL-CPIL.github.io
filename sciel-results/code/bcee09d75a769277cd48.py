"""Rewritten program for symbolic regression with 1 input."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers an unknown function using a sum of 5 Gaussian basis functions.
    
    The model is:
    y(x) = sum_{i=0}^{4} A_i * exp(-((x - mu_i)^2) / (2 * sigma^2))
    
    Parameters:
    p[0:5] -> Amplitudes A_i
    p[5:10] -> Means mu_i (with a small index-dependent offset to break symmetry)
    
    Width sigma is fixed at 1.0.
    """
    x = np.asarray(x, dtype=np.float64)
    p = params
    
    # Extract parameters
    amps = p[0:5]
    means_raw = p[5:10]
    
    # Add small offset to break symmetry of initial all-ones params
    # This allows BFGS to separate the Gaussians
    offsets = np.array([-0.4, -0.2, 0.0, 0.2, 0.4])
    means = means_raw + offsets
    
    # Fixed width
    sigma = 1.0
    sigma2 = 2.0 * sigma * sigma
    
    # Compute Gaussians
    # x shape: (N,)
    # means shape: (5,)
    # We need (N, 5) for broadcasting
    diff = x[:, np.newaxis] - means[np.newaxis, :]
    gaussians = np.exp(-(diff ** 2) / sigma2)
    
    # Linear combination
    # amps shape: (5,)
    result = gaussians.dot(amps)
    
    return result

# EVOLVE-BLOCK-END
