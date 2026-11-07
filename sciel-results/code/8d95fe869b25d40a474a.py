"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers an unknown function using a linear combination of Gaussian Radial Basis Functions.
    Basis: 10 Gaussians with fixed centers evenly spaced in [-3, 3] and fixed width.
    """
    x = np.asarray(x)
    
    # Define fixed centers for the RBFs
    centers = np.linspace(-3.0, 3.0, 10)
    sigma = 0.8  # Width parameter
    
    # Compute basis functions: exp(-(x - c_i)^2 / (2 * sigma^2))
    # x shape: (N,), centers shape: (10,)
    # diff shape: (N, 10)
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    
    # Compute the Gaussian terms
    # term shape: (N, 10)
    basis = np.exp(-0.5 * (diff / sigma) ** 2)
    
    # Linear combination: y = basis @ params
    # basis shape: (N, 10), params shape: (10,)
    y = basis @ params
    
    return y
# EVOLVE-BLOCK-END
