"""Kernel Ridge Regression with Gaussian RBF kernels for symbolic regression."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Model: linear combination of Gaussian RBF kernels with fixed centers.
    f(x) = sum_{i=0}^{9} w_i * exp(-gamma * (x - c_i)^2)
    
    where c_i are evenly spaced centers in [-3, 3] and gamma controls width.
    params (length 10) are the weights w_i.
    
    This provides local flexibility: each parameter controls contribution
    near its center, allowing the model to fit complex local structures
    that global basis functions (polynomials, Fourier) may struggle with.
    """
    # Fixed centers evenly spaced in [-3, 3]
    centers = np.linspace(-3.0, 3.0, 10)
    
    # Gaussian width parameter - chosen to give good local coverage
    # With 10 centers over range 6, spacing is 0.667
    # sigma ~ spacing/2 gives good overlap
    sigma = 0.5
    gamma = 1.0 / (2.0 * sigma ** 2)
    
    # Compute RBF features: shape (n_points, 10)
    # x has shape (n_points,)
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    rbf = np.exp(-gamma * diff ** 2)
    
    # Linear combination with weights
    # params has shape (10,)
    result = rbf @ params
    
    return result
# EVOLVE-BLOCK-END
