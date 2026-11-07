"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Non-linear RBF model with 10 fixed centers and a single learnable width.
    
    params[0:10] -> weights for 10 RBFs
    params[10]   -> log-width (so width = exp(params[10]))
    
    The 10 centers are evenly spaced over [-3, 3].
    prediction = sum_i w_i * exp(-((x - c_i)^2) / (2 * sigma^2))
    where sigma = exp(params[10])
    """
    # 10 evenly spaced centers over [-3, 3]
    centers = np.linspace(-3.0, 3.0, 10)
    
    weights = params[0:10]
    log_width = params[10]
    
    # Clamp log_width to avoid numerical issues
    log_width = np.clip(log_width, -5.0, 5.0)
    width = np.exp(log_width)
    
    # Compute RBF values: shape (n_points, 10)
    # x is shape (n,), centers is shape (10,)
    diff = x[:, np.newaxis] - centers[np.newaxis, :]  # (n, 10)
    rbf_values = np.exp(-(diff ** 2) / (2.0 * width ** 2))  # (n, 10)
    
    # prediction = rbf_values @ weights
    prediction = rbf_values @ weights
    
    return prediction
# EVOLVE-BLOCK-END
