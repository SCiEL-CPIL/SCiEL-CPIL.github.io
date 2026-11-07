"""Symbolic regression with 2 inputs using Gaussian RBF basis."""
import numpy as np
import time


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression using a Gaussian Radial Basis Function (RBF) network.
    
    The 10 parameters are linear weights for 10 fixed Gaussian basis functions
    centered at spatially distributed points in [-1, 1]^2. This local
    interpolation approach fundamentally differs from global polynomial bases,
    allowing the model to capture arbitrary non-smooth or highly local features.
    
    Centers (3x3 grid + origin + one extra for 10 total):
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),  (0, 0),  (0, 1),
    (1, -1),  (1, 0),  (1, 1),
    (0.5, 0.5)
    
    Shared bandwidth sigma = 0.5, chosen to balance locality and coverage.
    """
    t_start = time.time()
    TIME_LIMIT = 140.0

    # Ensure inputs are 1D arrays
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)

    # Define 10 fixed centers
    centers = np.array([
        [-1.0, -1.0],
        [-1.0,  0.0],
        [-1.0,  1.0],
        [ 0.0, -1.0],
        [ 0.0,  0.0],
        [ 0.0,  1.0],
        [ 1.0, -1.0],
        [ 1.0,  0.0],
        [ 1.0,  1.0],
        [ 0.5,  0.5],
    ], dtype=np.float64)

    # Shared bandwidth
    sigma = 0.5
    inv_2sigma2 = 1.0 / (2.0 * sigma * sigma)

    # Compute squared distances from each data point to each center
    # x, y have shape (n,)
    # centers have shape (10, 2)
    # dist_sq[i, j] = (x[i] - cx[j])^2 + (y[i] - cy[j])^2
    cx = centers[:, 0]  # shape (10,)
    cy = centers[:, 1]  # shape (10,)

    # Efficient computation: (n, 1) - (1, 10) -> (n, 10)
    dx = x[:, np.newaxis] - cx[np.newaxis, :]
    dy = y[:, np.newaxis] - cy[np.newaxis, :]
    dist_sq = dx * dx + dy * dy

    # Gaussian RBF values: exp(-dist_sq / (2 * sigma^2))
    basis = np.exp(-inv_2sigma2 * dist_sq)  # shape (n, 10)

    # Linear combination with the 10 parameters
    prediction = basis @ params

    # Time guard
    elapsed = time.time() - t_start
    if elapsed > TIME_LIMIT:
        return prediction

    return prediction
