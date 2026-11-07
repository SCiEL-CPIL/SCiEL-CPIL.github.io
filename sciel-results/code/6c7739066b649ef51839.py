"""Program for symbolic regression task using adaptive-width RBF network."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Adaptive-width RBF network with 5 fixed centers.
    
    Args:
        x: Input array of shape (N,).
        params: Array of 10 parameters.
            params[0:5] = Amplitudes for 5 Gaussian kernels
            params[5:10] = Log-widths (log-sigmas) for 5 Gaussian kernels
    
    Returns:
        Prediction array of shape (N,).
    """
    x = np.asarray(x, dtype=float)
    p = np.asarray(params, dtype=float)
    
    # Safety check for non-finite params
    if not np.all(np.isfinite(p)):
        return np.zeros_like(x)
    
    # Fixed centers evenly spaced over [-3, 3]
    centers = np.array([-3.0, -1.5, 0.0, 1.5, 3.0])
    
    # Extract amplitudes and log-widths
    amplitudes = p[0:5]
    log_sigmas = p[5:10]
    
    # Convert log-sigmas to sigmas, clamp to reasonable range to avoid overflow/underflow
    sigmas = np.exp(np.clip(log_sigmas, -2.0, 2.0))  # sigma in [~0.135, ~7.39]
    
    # Compute RBF values
    # diff shape: (N, 5)
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    
    # sigma_sq_2 shape: (5,)
    sigma_sq_2 = 2.0 * sigmas**2
    
    # rbf_values shape: (N, 5)
    rbf_values = np.exp(-(diff**2) / sigma_sq_2[np.newaxis, :])
    
    # Weighted sum
    result = rbf_values @ amplitudes
    
    return result

# EVOLVE-BLOCK-END
