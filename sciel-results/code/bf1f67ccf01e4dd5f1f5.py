# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_TIMEOUT_LIMIT = 110  # Safety margin before 120s limit

# Fixed centers for Radial Basis Functions
# Covering the domain [-1, 1] x [-1, 1]
_CENTERS = np.array([
    [0.0, 0.0],
    [0.5, 0.0],
    [-0.5, 0.0],
    [0.0, 0.5],
    [0.0, -0.5],
    [0.5, 0.5],
    [-0.5, 0.5],
    [0.5, -0.5],
    [-0.5, -0.5],
    [0.8, 0.8]
])
_SIGMA = 0.5  # Fixed width for RBFs

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression with 2 inputs x, y in [-1, 1]^2.
    
    Strategy: Radial Basis Function (RBF) expansion.
    Uses 10 fixed centers distributed across the domain.
    The 10 parameters are the weights for each RBF.
    RBFs provide a flexible local approximation capability,
    often outperforming global bases like Fourier or low-degree
    polynomials for general smooth functions.
    """
    # Deadline guard
    if time.time() - _start_time > _TIMEOUT_LIMIT:
        return np.full_like(y, np.mean(y))
        
    p = params
    
    # Calculate squared distances from each point (x, y) to each center
    # x, y are arrays of shape (N,)
    # _CENTERS is shape (10, 2)
    
    # Efficient computation:
    # x[:, None] - _CENTERS[:, 0] -> (N, 10)
    # y[:, None] - _CENTERS[:, 1] -> (N, 10)
    dx = x[:, None] - _CENTERS[None, :, 0]
    dy = y[:, None] - _CENTERS[None, :, 1]
    
    # Squared distances
    dist_sq = dx**2 + dy**2
    
    # RBF values: exp(-dist_sq / (2 * sigma^2))
    # Shape: (N, 10)
    rbf_values = np.exp(-dist_sq / (2 * _SIGMA**2))
    
    # Linear combination: (N, 10) @ (10,) -> (N,)
    out = rbf_values @ p
    
    return out
# EVOLVE-BLOCK-END
