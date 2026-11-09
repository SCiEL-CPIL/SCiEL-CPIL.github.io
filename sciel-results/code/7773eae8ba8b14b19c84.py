import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Fixed centers for 10 Gaussian basis functions
    # Distributed evenly across the input domain [-3, 3]
    centers = np.linspace(-3.0, 3.0, 10)
    
    # Fixed bandwidth (sigma)
    sigma = 1.0
    
    # Compute the RBF basis matrix
    # x shape: (N,)
    # centers shape: (10,)
    # We want to compute exp(-(x_i - c_j)^2 / (2 * sigma^2)) for all i, j
    
    # Reshape x for broadcasting
    x_col = x[:, np.newaxis]
    c_row = centers[np.newaxis, :]
    
    # Compute squared distances
    dist_sq = (x_col - c_row) ** 2
    
    # Compute RBF values
    # Shape: (N, 10)
    basis = np.exp(-dist_sq / (2 * sigma**2))
    
    # Combine with parameters
    # params shape: (10,)
    # y shape: (N,)
    y = basis @ params
    
    return y
