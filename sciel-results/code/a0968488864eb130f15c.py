"""Symbolic regression with RBF basis using 10 learnable parameters."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Universal approximator using 5 Gaussian Radial Basis Functions.
    
    Parameters (10 total):
    - params[0:5]: centers c_1, ..., c_5
    - params[5:10]: amplitudes a_1, ..., a_5
    
    Widths are fixed at sigma = 0.8 to keep the problem well-conditioned
    while still allowing localized bumps to capture non-linear structure.
    
    The model: f(x) = sum_i a_i * exp(-(x - c_i)^2 / (2 * sigma^2))
    
    This is a smooth, differentiable function suitable for BFGS optimization.
    Starting from all-ones, the optimizer can adaptively place and weight
    the RBFs to match the hidden target function.
    """
    sigma = 0.8
    centers = params[0:5]
    amplitudes = params[5:10]
    
    # Compute RBF contributions
    # x has shape (n,), centers has shape (5,)
    # We need to compute for each i: (x - centers[i])^2
    # Result should be shape (n,)
    
    # Efficient computation using broadcasting
    # x[:, None] has shape (n, 1), centers[None, :] has shape (1, 5)
    x_expanded = x[:, np.newaxis]  # (n, 1)
    centers_expanded = centers[np.newaxis, :]  # (1, 5)
    
    # Squared distances: (n, 5)
    sq_dist = (x_expanded - centers_expanded) ** 2
    
    # RBF values: (n, 5)
    rbf_values = np.exp(-sq_dist / (2.0 * sigma ** 2))
    
    # Weighted sum: (n, 5) @ (5,) -> (n,)
    result = rbf_values @ amplitudes
    
    return result
# EVOLVE-BLOCK-END
