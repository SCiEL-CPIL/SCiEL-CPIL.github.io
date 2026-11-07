"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).ravel()
    
    # Ensure params is a flat array of length 10
    params = np.asarray(params, dtype=np.float64).ravel()
    if len(params) < 10:
        # Pad with zeros if necessary, though contract says length 10
        params = np.pad(params, (0, 10 - len(params)))
    else:
        params = params[:10]
        
    # Define 10 fixed centers evenly spaced in [-2, 2]
    # This provides local flexibility to approximate diverse shapes
    centers = np.linspace(-2.0, 2.0, 10)
    
    # Fixed width for the Gaussian kernels
    # A width of 0.5 provides a good balance between local and global influence
    # over the interval [-2, 2]
    width = 0.5
    
    # Compute the RBF basis functions
    # We need to compute exp(-((x - c_i)^2) / (2 * width^2)) for each center c_i
    # x is shape (N,), centers is shape (10,)
    # We want a matrix of shape (N, 10)
    
    # Efficient computation using broadcasting
    # x[:, np.newaxis] has shape (N, 1)
    # centers[np.newaxis, :] has shape (1, 10)
    # diff has shape (N, 10)
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    
    # Calculate squared distances
    diff_sq = diff ** 2
    
    # Calculate Gaussian values
    # Gaussian: exp(-diff_sq / (2 * width^2))
    inv_var = 1.0 / (2.0 * width ** 2)
    rbf_values = np.exp(-diff_sq * inv_var)
    
    # The prediction is the weighted sum of the RBF values
    # rbf_values is (N, 10), params is (10,)
    # Result is (N,)
    prediction = rbf_values @ params
    
    return prediction
# EVOLVE-BLOCK-END
