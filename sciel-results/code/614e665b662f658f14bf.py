"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Gaussian RBF Network with learnable centers.
    
    params[0:5]  : Centers for the 5 Gaussian kernels
    params[5:10] : Amplitudes for the 5 Gaussian kernels
    
    The width (sigma) is fixed at 0.5. This design allows the model to
    adapt the location of basis functions to the data, capturing local
    features more effectively than fixed centers.
    """
    # Extract centers and amplitudes
    centers = params[0:5]
    amplitudes = params[5:10]
    
    # Reshape for broadcasting
    # x is shape (N,), centers is shape (5,)
    x_col = x.reshape(-1, 1)
    c_row = centers.reshape(1, -1)
    
    # Compute squared distances
    sq_dist = (x_col - c_row) ** 2
    
    # Fixed width sigma = 0.5, so 2 * sigma^2 = 0.5
    # This provides a good balance between local flexibility and smoothness
    kernels = np.exp(-sq_dist / 0.5)
    
    # Weighted sum of kernels
    rbf_part = kernels @ amplitudes
    
    return rbf_part
# EVOLVE-BLOCK-END
