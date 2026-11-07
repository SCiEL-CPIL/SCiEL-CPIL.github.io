"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    RBF-based symbolic regression with 10 fixed centers and a single learnable width.
    
    params[0] is the width (sigma) of the RBFs.
    params[1:11] are the weights for the 10 RBFs.
    
    To ensure stability and well-conditioning for BFGS:
    - We use fixed, evenly spaced centers in [-3, 3].
    - The width is derived from params[0] using a softplus-like or absolute value 
      transformation to keep it positive, but since BFGS starts at 1 and we want 
      smoothness, we use a scaled exponential or just a direct positive mapping.
      Actually, the directive says "single learnable width parameter". 
      Let's use params[0] directly as the width, but ensure it's positive.
      Since BFGS can go negative, we should map it. 
      However, if we just use abs(params[0]) + 0.1, it's non-smooth at 0.
      Better: use np.exp(params[0]) * scale or just params[0] if we assume it stays positive.
      Given the start is 1, and BFGS is a quasi-Newton method, it might stay positive 
      if the landscape is smooth. But to be safe, let's use a smooth positive mapping.
      
      Let's use: sigma = np.exp(params[0]) * 0.5 + 0.1
      This ensures sigma > 0.1 and is smooth.
      
    Centers: 10 evenly spaced points in [-3, 3].
    c_i = -3 + (6 * i) / 9 for i in 0..9
    """
    # Extract parameters
    sigma = np.exp(params[0]) * 0.5 + 0.1  # Smoothly positive width
    weights = params[1:11]
    
    # Define fixed centers
    # 10 centers evenly spaced from -3 to 3
    # np.linspace(-3, 3, 10)
    centers = np.linspace(-3, 3, 10)
    
    # Compute RBF values
    # x is shape (N,), centers is shape (10,)
    # We want a matrix of shape (N, 10) where M[i, j] = exp(-((x_i - c_j)^2) / (2 * sigma^2))
    
    # Reshape for broadcasting
    x_col = x.reshape(-1, 1)  # (N, 1)
    c_row = centers.reshape(1, -1)  # (1, 10)
    
    # Compute squared distances
    diff = x_col - c_row  # (N, 10)
    sq_diff = diff ** 2
    
    # Compute RBF values
    rbf_values = np.exp(-sq_diff / (2.0 * sigma * sigma))  # (N, 10)
    
    # Linear combination
    prediction = rbf_values @ weights  # (N,)
    
    return prediction
# EVOLVE-BLOCK-END
