"""Symbolic regression with 1 input; RBF kernel basis for local smooth function fitting."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Model using Radial Basis Functions (Gaussian kernels) with fixed centers.
    
    y = sum_{i=0}^{9} exp(params_i) * exp(-(x - c_i)^2 / (2 * sigma^2))
    
    Centers c_i are evenly spaced across the domain [0.1, 6].
    Using exp(params_i) ensures positive weights, improving numerical stability
    and allowing BFGS to optimize over unconstrained parameter space.
    
    The local nature of RBF kernels allows fitting arbitrary smooth functions
    by adapting weights to local function behavior.
    """
    x = np.asarray(x, dtype=float)
    
    # Define 10 evenly spaced centers across [0.1, 6]
    x_min, x_max = 0.1, 6.0
    centers = np.linspace(x_min, x_max, 10)
    
    # Fixed bandwidth: approximately 1/10 of the domain width
    # Domain width = 5.9, so sigma ≈ 0.59
    sigma = 0.59
    
    # Compute RBF values: shape (N, 10)
    # For each x_j and center c_i: exp(-(x_j - c_i)^2 / (2 * sigma^2))
    x_col = x[:, np.newaxis]  # (N, 1)
    centers_row = centers[np.newaxis, :]  # (1, 10)
    
    diff = x_col - centers_row  # (N, 10)
    rbf_values = np.exp(-diff ** 2 / (2.0 * sigma ** 2))  # (N, 10)
    
    # Use exp(params) to ensure positive weights
    # params is shape (10,), so weights shape (10,)
    weights = np.exp(params)  # (10,)
    
    # Prediction: sum_i weights_i * rbf_values[:, i]
    # rbf_values is (N, 10), weights is (10,)
    y = rbf_values @ weights  # (N,)
    
    return y
# EVOLVE-BLOCK-END
