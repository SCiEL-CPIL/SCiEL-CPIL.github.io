"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Kernel Ridge Regression style model.
    Uses 10 Gaussian RBF kernels with fixed centers evenly spaced in [-3, 3].
    The 10 params are the weights for these kernels.
    """
    x = np.asarray(x, dtype=float)
    p = np.asarray(params, dtype=float)
    
    # Ensure we have at least 10 params, pad with zeros if necessary
    if p.shape[0] < 10:
        p = np.pad(p, (0, 10 - p.shape[0]), mode='constant')
    
    # Fixed centers for the RBF kernels
    # Spaced from -2.7 to 2.7 to avoid edge effects, or just -3 to 3
    # Let's use 10 centers evenly spaced in [-3, 3]
    centers = np.linspace(-3.0, 3.0, 10)
    
    # Bandwidth (sigma) for the RBF kernels
    # A moderate bandwidth allows local flexibility while maintaining some smoothness
    # If sigma is too small, it becomes interpolation (prone to overfitting/noise).
    # If sigma is too large, it becomes a global basis (like current poor performance).
    # Let's try sigma = 0.5 or 1.0. 
    # Distance between centers is 6/9 = 0.66. 
    # Sigma = 0.5 means kernels overlap significantly but are distinct.
    sigma = 0.5
    
    # Compute RBF values: K(x, c_i) = exp(-||x - c_i||^2 / (2 * sigma^2))
    # x shape: (N,)
    # centers shape: (10,)
    # diff shape: (N, 10)
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    
    # RBF matrix
    K = np.exp(-diff**2 / (2 * sigma**2))
    
    # Prediction is K @ p
    # K shape: (N, 10), p shape: (10,)
    result = K @ p
    
    return result
# EVOLVE-BLOCK-END
