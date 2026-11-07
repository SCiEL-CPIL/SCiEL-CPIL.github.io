"""Program for symbolic regression task using a flexible basis model."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Flexible model combining:
    - Polynomial basis (global smooth trend)
    - Gaussian RBFs (local flexibility)
    - Periodic terms (capture oscillatory behavior)
    
    Args:
        x: Input array of shape (N,).
        params: Array of 10 parameters.
            params[0:4] = Polynomial coefficients x^0 to x^3
            params[4:7] = RBF amplitudes for 3 centers
            params[7:9] = Sine and cosine amplitudes (frequency 1.0)
            params[9] = Linear term coefficient
    
    Returns:
        Prediction array of shape (N,).
    """
    x = np.asarray(x, dtype=float)
    p = np.asarray(params, dtype=float)
    
    # Safety check for non-finite params
    if not np.all(np.isfinite(p)):
        return np.zeros_like(x)
    
    # Polynomial basis: 1, x, x^2, x^3
    poly_part = p[0] + p[1] * x + p[2] * x**2 + p[3] * x**3
    
    # RBF part with 3 centers
    centers = np.array([-1.5, 0.0, 1.5])
    sigma = 1.0
    sigma_sq_2 = 2.0 * sigma**2
    
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    rbf_values = np.exp(-(diff**2) / sigma_sq_2)
    rbf_part = rbf_values @ p[4:7]
    
    # Periodic terms
    periodic_part = p[7] * np.sin(x) + p[8] * np.cos(x)
    
    # Linear term
    linear_part = p[9] * x
    
    result = poly_part + rbf_part + periodic_part + linear_part
    
    return result

# EVOLVE-BLOCK-END
