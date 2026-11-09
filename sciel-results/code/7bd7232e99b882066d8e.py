"""RBF-based symbolic regression model with adaptive widths."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    RBF model with 5 fixed centers and adaptive widths.
    
    params[0:5]   -> weights for each RBF
    params[5:10]  -> log-widths (log_sigma) for each RBF, exp-transformed
    
    Centers are fixed at evenly spaced points in [-3, 3].
    """
    x = np.asarray(x, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    
    # 5 fixed centers evenly spaced in [-2.5, 2.5] to better cover the interior
    centers = np.linspace(-2.5, 2.5, 5)
    
    weights = p[0:5]
    # Use exp on the width parameters to ensure positive widths
    log_sigma = np.clip(p[5:10], -5, 5) # Tighter clip to prevent extreme widths
    sigma = np.exp(log_sigma)
    # Ensure a minimum width to avoid numerical instability (spikes)
    sigma = np.clip(sigma, 0.1, 10.0)
    
    # Compute IMQ (Inverse Multiquadric) RBF values
    # IMQ: 1 / (1 + ((x - center) / sigma)^2)
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    scaled_diff = diff / sigma[np.newaxis, :]
    
    rbf_values = 1.0 / (1.0 + scaled_diff ** 2)
    
    # Weighted sum
    prediction = rbf_values @ weights
    
    return prediction
# EVOLVE-BLOCK-END
