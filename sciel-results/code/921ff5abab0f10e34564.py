# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Non-linear parametric model: Gaussian Mixture Model (GMM) with 3 components.
    
    Parameters (10 total):
    - p[0:3] : Means (mu_1, mu_2, mu_3)
    - p[3:6] : Widths (sigma_1, sigma_2, sigma_3) - clamped to avoid division by zero
    - p[6:9] : Weights (w_1, w_2, w_3)
    - p[9]   : Global offset (bias)
    
    The model is: y = bias + sum_{i=1}^{3} w_i * exp(-0.5 * ((x - mu_i) / sigma_i)^2)
    
    This is non-linear in parameters (mu, sigma, w interact), allowing the optimizer
    to shift and reshape the basis functions to better fit the target, unlike the
    linear basis approach where only coefficients change.
    """
    x = np.asarray(x, dtype=np.float64)
    orig_shape = x.shape
    x_flat = x.ravel()
    
    p = params
    
    # Extract parameters
    mu = p[0:3]
    sigma = p[3:6]
    w = p[6:9]
    bias = p[9]
    
    # Clamp sigma to prevent numerical instability (division by very small numbers)
    # A small positive value ensures stability without restricting the fit too much
    sigma_safe = np.clip(sigma, 1e-3, None)
    
    # Compute Gaussian components
    # (N, 3) matrix of Gaussian values
    # x_flat is (N,), mu is (3,) -> broadcasting to (N, 3)
    diff = x_flat[:, np.newaxis] - mu[np.newaxis, :]
    gaussians = np.exp(-0.5 * (diff / sigma_safe[np.newaxis, :]) ** 2)
    
    # Weighted sum
    # gaussians (N, 3) * w (3,) -> (N,)
    result = np.sum(gaussians * w[np.newaxis, :], axis=1) + bias
    
    return result.reshape(orig_shape)
# EVOLVE-BLOCK-END
