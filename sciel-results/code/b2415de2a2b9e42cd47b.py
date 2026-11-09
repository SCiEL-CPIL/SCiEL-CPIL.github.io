"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

# Fixed hyperparameters for the RBF basis
# Domain is [0.1, 6]
# We use 10 RBF centers evenly spaced to utilize all 10 parameters
_CENTERS = np.linspace(0.1, 6.0, 10)
# Fixed bandwidth
# Slightly wider bandwidth to encourage smoother fits and better generalization
_SIGMA = 0.6
# Pre-compute the denominator for the exponent to save time
_INV_2_SIGMA2 = 1.0 / (2.0 * _SIGMA * _SIGMA)


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Computes the model prediction.
    
    Model: y = sum(p[i] * exp(-(x-c_i)^2 / (2*sigma^2)))
    All 10 parameters are RBF weights.
    
    Args:
    x: Input array of shape (N,)
    params: Parameter array of shape (10,)
        - params[0:10]: RBF weights
    
    Returns:
    Prediction array of shape (N,)
    """
    # Ensure x is a 1D array for calculation, though it might come as column
    x_flat = np.asarray(x).flatten()
    
    # params mapping
    w_rbf = params
    
    # Compute RBF values
    # x_flat shape: (N,)
    # _CENTERS shape: (10,)
    # diff shape: (N, 10)
    diff = x_flat[:, np.newaxis] - _CENTERS[np.newaxis, :]
    
    # Exponent part
    # exp_arg shape: (N, 10)
    exp_arg = -diff * diff * _INV_2_SIGMA2
    
    # RBF matrix shape: (N, 10)
    rbf_matrix = np.exp(exp_arg)
    
    # Linear combination
    # rbf_matrix @ w_rbf -> shape (N,)
    y = rbf_matrix @ w_rbf
    
    return y
# EVOLVE-BLOCK-END
