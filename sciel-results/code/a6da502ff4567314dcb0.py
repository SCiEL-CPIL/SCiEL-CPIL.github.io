"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Radial Basis Function (RBF) basis with fixed centers.
    # Linear model in params, robust for BFGS.
    # 10 centers evenly spaced in [-3, 3].
    centers = np.linspace(-3.0, 3.0, 10)
    sigma = 0.5  # Smaller sigma allows for more local features
    
    # x is expected to be a 1D array of inputs
    # Ensure x is a column vector for broadcasting
    x_col = x.reshape(-1, 1)
    
    # Compute basis functions
    # diff shape: (N, 10)
    diff = x_col - centers
    # basis shape: (N, 10)
    basis = np.exp(-0.5 * (diff / sigma) ** 2)
    
    # Prediction is the linear combination of basis functions with weights params
    return np.einsum('i,ij->j', params, basis)
# EVOLVE-BLOCK-END
