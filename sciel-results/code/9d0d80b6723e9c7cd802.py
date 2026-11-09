"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Approximate the target function using a RBF (Radial Basis Function) network
    with Gaussian kernels and fixed centers spanning the input domain.
    
    Args:
        x: Input array of shape (N,).
        params: Array of 10 parameters.
        
    Returns:
        Prediction array of shape (N,).
    """
    x = np.asarray(x, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)
    
    # Fixed centers evenly spaced in [0.1, 6]
    n_centers = 8
    x_min = 0.1
    x_max = 6.0
    centers = np.linspace(x_min, x_max, n_centers)
    
    # Learnable width parameter: use a softplus-like transformation to ensure positivity
    # width = exp(params[0]) * scale + offset
    # We use a robust parameterization: width = 0.1 + exp(params[0])
    # This ensures width > 0.1, avoiding numerical issues with very small widths
    width = 0.1 + np.exp(np.clip(params[0], -10, 10))
    
    # Compute Gaussian RBF values: phi_i(x) = exp(-((x - c_i)^2) / (2 * width^2))
    # Shape: (N, n_centers)
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    rbf_values = np.exp(-(diff ** 2) / (2.0 * width ** 2))
    
    # Remaining 9 parameters are weights for the 8 RBFs + a bias term
    # params[1:9] are weights for the 8 RBFs
    # params[9] is a bias term
    
    weights = params[1:9]
    bias = params[9]
    
    # Prediction: sum of weighted RBFs + bias
    prediction = rbf_values @ weights + bias
    
    return prediction
# EVOLVE-BLOCK-END
