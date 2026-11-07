"""
Strategy: Universal approximation via a fixed RBF basis.
The evaluator fits 10 parameters (weights) using BFGS.
We define a basis of 10 functions:
- 1 constant (intercept)
- 9 RBFs centered at a 3x3 grid in [-1, 1]^2.

This allows the model to approximate a wide variety of smooth bivariate functions.
The basis is precomputed for efficiency.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Precompute the RBF centers and basis matrix structure?
# No, x and y change every call. But we can precompute the centers.

# Centers for the 3x3 grid
centers = []
for cx in [-1.0, 0.0, 1.0]:
    for cy in [-1.0, 0.0, 1.0]:
        centers.append((cx, cy))
centers = np.array(centers)  # shape (9, 2)

# We have 10 params.
# param[0] is the weight for the constant term (1.0)
# param[1] to param[9] are weights for the 9 RBFs.

# RBF width (sigma). A fixed width is safer for BFGS stability.
# If sigma is too small, the RBFs are too local (underfitting on sparse grid).
# If sigma is too large, they are too smooth (overfitting/underfitting depending on function).
# For inputs in [-1, 1], a width of 0.5 to 1.0 is reasonable.
# Let's use sigma = 0.8.
SIGMA = 0.8
SIGMA_SQ = 2 * SIGMA * SIGMA  # For exp(-d^2 / (2 sigma^2))


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    x: array of shape (N, 2) or (N,) if 1D? Task says 2 inputs, so likely (N, 2) or two arrays?
    Task says: "equation(x, y, params)". Usually x is the input features.
    "2 input(s)" implies x might be a tuple (x1, x2) or an array of shape (N, 2).
    Looking at the seed: `params[0] * x + params[1] * y`. This suggests x and y are separate arrays of shape (N,).
    Wait, the seed uses `x` and `y` as if they are the two input variables.
    "Data: 200 training points ... inputs drawn uniformly from [-1, 1], [-1, 1]."
    "equation(x, y, params)".
    In the seed: `return params[0] * x + params[1] * y + params[2]`.
    This confirms x and y are 1D arrays of length N, representing the two input dimensions.
    """
    
    # x: shape (N,)
    # y: shape (N,)
    # params: shape (10,)
    
    # Construct the design matrix for the basis functions
    # Basis 0: Constant 1
    # Basis 1-9: RBFs
    
    # Calculate squared distances from each point to each center
    # x[:, None] is (N, 1), centers[None, :, 0] is (1, 9)
    # dist_sq = (x - cx)^2 + (y - cy)^2
    
    dx = x[:, None] - centers[None, :, 0]  # (N, 9)
    dy = y[:, None] - centers[None, :, 1]  # (N, 9)
    
    dist_sq = dx**2 + dy**2  # (N, 9)
    
    # RBF values: exp(-dist_sq / (2 * sigma^2))
    rbf_values = np.exp(-dist_sq / SIGMA_SQ)  # (N, 9)
    
    # Prepend the constant term
    # constant term is 1 for all points
    ones = np.ones((x.shape[0], 1))
    basis = np.hstack((ones, rbf_values))  # (N, 10)
    
    # Prediction: basis @ params
    return basis @ params
# EVOLVE-BLOCK-END
