"""Symbolic regression with 3 inputs using an RBF (radial basis function) basis."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Fixed centers for the RBFs, spread across the domain [0.5, 2]^3
# We need exactly 10 centers to match the 10 parameters.
# Use a structured grid: 2x2x2 = 8 centers, plus 2 more.
# Actually, let's use 10 well-distributed points.
# A simple approach: use a 2x2x2 grid (8 points) plus 2 face centers.
# Grid points at {0.5, 2} for each coordinate:
# (0.5,0.5,0.5), (0.5,0.5,2), (0.5,2,0.5), (0.5,2,2),
# (2,0.5,0.5), (2,0.5,2), (2,2,0.5), (2,2,2)
# Plus two more: center of domain (1.25,1.25,1.25) and another.
# Let's use a more systematic set:
# 10 points: a 2x2x2 grid plus 2 additional points.

_centers = np.array([
    [0.5, 0.5, 0.5],
    [0.5, 0.5, 2.0],
    [0.5, 2.0, 0.5],
    [0.5, 2.0, 2.0],
    [2.0, 0.5, 0.5],
    [2.0, 0.5, 2.0],
    [2.0, 2.0, 0.5],
    [2.0, 2.0, 2.0],
    [1.25, 1.25, 1.25],
    [1.25, 0.5, 1.25],
], dtype=np.float64)

# RBF width (sigma). Smaller sigma = more localized.
# The domain extent is about 1.5 in each dimension, so sigma ~ 0.5-1.0 is reasonable.
_SIGMA = 0.75


def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    RBF-based closed-form expression.
    
    params: array of length 10, weights for the 10 RBFs.
    Returns: array with one prediction per point.
    """
    start_time = time.time()
    deadline = start_time + 100.0  # Safety margin before 120s limit
    
    # Build input matrix: (N, 3)
    N = x.shape[0]
    inputs = np.column_stack((x, y, z))
    
    # Compute RBF activations: (N, 10)
    # For each point and each center, compute exp(-||input - center||^2 / (2*sigma^2))
    # Vectorized: inputs is (N,3), centers is (10,3)
    # diff = inputs[:, np.newaxis, :] - centers[np.newaxis, :, :]  # (N, 10, 3)
    # sq_dist = np.sum(diff**2, axis=2)  # (N, 10)
    # activations = np.exp(-sq_dist / (2 * _SIGMA**2))  # (N, 10)
    
    # More memory-efficient approach: compute in chunks if N is large
    # But N is only 200, so full vectorization is fine.
    diff = inputs[:, np.newaxis, :] - _centers[np.newaxis, :, :]
    sq_dist = np.sum(diff * diff, axis=2)
    activations = np.exp(-sq_dist / (2.0 * _SIGMA * _SIGMA))
    
    # Prediction: activations @ params  # (N,)
    pred = activations @ params
    
    # Ensure finite output
    if not np.all(np.isfinite(pred)):
        # Fallback: return zeros if something went wrong
        pred = np.zeros(N, dtype=np.float64)
    
    return pred
# EVOLVE-BLOCK-END
