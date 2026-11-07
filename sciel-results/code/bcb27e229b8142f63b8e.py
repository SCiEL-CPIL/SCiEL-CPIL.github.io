"""Gaussian RBF basis with 10 parameters: 5 centers, 2 widths, 3 global terms."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Approximates the target function using a Gaussian RBF basis plus global terms.
    
    The 10 parameters define:
    - params[0:5]: weights for 5 Gaussian RBFs at fixed centers
    - params[5:7]: width parameters (inverse of sigma^2)
    - params[7:10]: coefficients for global basis [1, x, x^2]
    
    y(x) = sum_{i=0}^{4} w_i * exp(-alpha * (x - c_i)^2) + p7 + p8*x + p9*x^2
    where c_i are fixed centers at -2, -1, 0, 1, 2
    and alpha = params[5] * params[6] (product of two width params for flexibility)
    """
    n = len(x)
    if n == 0:
        return np.array([], dtype=float)
    
    x = np.asarray(x, dtype=float)
    
    # Fixed centers spread across [-2, 2]
    centers = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    
    # Extract parameters
    weights = params[0:5]
    alpha = params[5] * params[6]  # Product gives a single effective width parameter
    # Ensure alpha is positive for stable Gaussians
    if alpha < 0:
        alpha = 0.0
    
    # Global terms
    p7, p8, p9 = params[7], params[8], params[9]
    
    # Compute RBF contributions
    # x has shape (n,), centers has shape (5,)
    # We want a (n, 5) matrix of distances
    diff = x[:, np.newaxis] - centers[np.newaxis, :]  # (n, 5)
    
    # Compute Gaussian values
    if alpha > 1e-10:
        rbf_vals = np.exp(-alpha * diff ** 2)  # (n, 5)
    else:
        # If alpha is near zero, all Gaussians are ~1, so sum of weights
        rbf_vals = np.ones((n, 5))
    
    # Weighted sum of RBFs
    rbf_sum = rbf_vals @ weights  # (n,)
    
    # Add global terms
    result = rbf_sum + p7 + p8 * x + p9 * x ** 2
    
    return result
# EVOLVE-BLOCK-END
