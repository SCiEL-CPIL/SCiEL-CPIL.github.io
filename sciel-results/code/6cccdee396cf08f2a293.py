"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Structural change: Use a localized basis of Gaussian RBFs with fixed centers
    and one bias term. This is well-conditioned for BFGS and can approximate
    many smooth functions on [-2, 2].
    
    params[0] = bias
    params[1:10] = weights for 9 Gaussian RBFs centered at -2, -1.5, ..., 2
    """
    x = np.asarray(x, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)
    
    # Fixed centers evenly spaced in [-2, 2]
    centers = np.linspace(-2.0, 2.0, 9)
    # Fixed width - moderate to balance locality and coverage
    sigma = 1.0
    
    # Compute RBF values: shape (n_points, 9)
    # (x - c)^2 / (2*sigma^2)
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    rbf_values = np.exp(-0.5 * (diff / sigma) ** 2)
    
    # result = bias + rbf_values @ weights
    bias = params[0]
    weights = params[1:10]
    
    result = bias + rbf_values @ weights
    
    # Ensure finite
    if not np.all(np.isfinite(result)):
        result = np.zeros_like(x)
    
    return result
# EVOLVE-BLOCK-END
