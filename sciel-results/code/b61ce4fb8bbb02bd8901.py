"""Program for symbolic regression task using adaptive-width RBF with polynomial trend."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid model combining:
    - Low-degree polynomial trend (global smooth behavior)
    - Adaptive-width Gaussian RBFs (local flexibility)
    
    Args:
        x: Input array of shape (N,).
        params: Array of 10 parameters.
            params[0:3] = Polynomial coefficients for 1, x, x^2
            params[3:8] = Amplitudes for 5 Gaussian RBF kernels
            params[8:10] = Log-widths for the first 2 RBF kernels (others use default)
    
    Returns:
        Prediction array of shape (N,).
    """
    x = np.asarray(x, dtype=float)
    p = np.asarray(params, dtype=float)
    
    # Safety check for non-finite params
    if not np.all(np.isfinite(p)):
        return np.zeros_like(x)
    
    # Polynomial trend: constant, linear, quadratic
    # This captures the global trend and helps with extrapolation
    poly_part = p[0] + p[1] * x + p[2] * x**2
    
    # RBF part with 5 fixed centers
    centers = np.array([-3.0, -1.5, 0.0, 1.5, 3.0])
    
    # Extract amplitudes for all 5 RBFs
    amplitudes = p[3:8]
    
    # Extract log-widths for first 2 RBFs, use default for others
    # This allows adaptive width where it matters most while keeping stability
    log_sigmas = np.zeros(5)
    log_sigmas[0] = p[8]
    log_sigmas[1] = p[9]
    
    # Convert log-sigmas to sigmas, clamp to reasonable range
    sigmas = np.exp(np.clip(log_sigmas, -2.0, 2.0))  # sigma in [~0.135, ~7.39]
    
    # Compute RBF values
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    sigma_sq_2 = 2.0 * sigmas**2
    rbf_values = np.exp(-(diff**2) / sigma_sq_2[np.newaxis, :])
    
    # Weighted sum of RBFs
    rbf_part = rbf_values @ amplitudes
    
    result = poly_part + rbf_part
    
    # Safety: ensure finite output
    if not np.all(np.isfinite(result)):
        return np.zeros_like(x)
    
    return result

# EVOLVE-BLOCK-END
