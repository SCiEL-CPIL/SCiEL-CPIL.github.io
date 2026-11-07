"""Symbolic regression with Gaussian RBF basis for BFGS fitting."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Gaussian RBF basis with 10 parameters.
    
    Uses 10 Gaussian radial basis functions centered evenly across the input
    domain [0.1, 6]. Each basis function is exp(-((x - c_i)^2) / (2 * sigma^2)).
    
    The parameters are fitted by BFGS starting from ones. The RBF basis is
    well-conditioned and can represent arbitrary smooth functions, avoiding
    the conditioning issues of high-degree polynomials.
    
    params[0:10] are weights for the 10 RBF centers.
    """
    # Ensure x is a 1D array
    x = np.asarray(x, dtype=np.float64)
    
    # Define RBF centers evenly spaced in [0.1, 6]
    # 10 centers
    x_min = 0.1
    x_max = 6.0
    centers = np.linspace(x_min, x_max, 10)
    
    # Fixed width: approximately half the spacing between centers
    # Spacing is (6.0 - 0.1) / 9 ≈ 0.656
    # Use sigma = 0.4 for good overlap but distinct peaks
    sigma = 0.4
    
    # Compute RBF basis: shape (n_points, 10)
    # diff[i, j] = x_i - centers_j
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    
    # RBF: exp(-diff^2 / (2 * sigma^2))
    basis = np.exp(-diff**2 / (2.0 * sigma**2))
    
    # params is shape (10,)
    params = np.asarray(params, dtype=np.float64).reshape(-1)
    
    # Predict: (n, 10) @ (10,) -> (n,)
    pred = basis @ params
    
    return pred
# EVOLVE-BLOCK-END
