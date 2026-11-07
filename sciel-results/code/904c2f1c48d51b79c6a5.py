"""Symbolic regression for E. coli growth rate - RBF interpolant."""
import numpy as np

# Precomputed centers and gamma
# 4 corners
# 1 centroid
# 5 mid-edges (chosen for coverage)
_CENTERS = np.array([
    [0.0, 0.0, 0.0, 0.0],
    [1.0, 1.0, 1.0, 1.0],
    [0.0, 1.0, 0.0, 1.0],
    [1.0, 0.0, 1.0, 0.0],
    [0.5, 0.5, 0.5, 0.5],
    [0.5, 0.0, 0.0, 0.0],
    [0.5, 1.0, 1.0, 1.0],
    [0.0, 0.5, 0.0, 0.0],
    [1.0, 0.5, 1.0, 1.0],
    [0.0, 0.0, 0.5, 0.0]
])
_GAMMA = 2.0

def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    RBF interpolant with 10 fixed centers and 10 fitted weights.
    """
    # Normalize inputs to [0, 1]
    b_norm = (b - 1.0) / 4.0
    s_norm = (s - 1.0) / 39.0
    temp_norm = (temp - 1.0) / 11.0
    ph_norm = (pH - 1.0) / 4.0
    
    # Stack inputs into (N, 4)
    X = np.stack([b_norm, s_norm, temp_norm, ph_norm], axis=-1)
    
    # Compute distances to centers
    # X shape: (N, 4), Centers shape: (10, 4)
    # Diff shape: (N, 10, 4)
    diff = X[:, np.newaxis, :] - _CENTERS[np.newaxis, :, :]
    
    # Squared distances
    sq_dist = np.sum(diff ** 2, axis=-1)
    
    # RBF values
    rbf_values = np.exp(-_GAMMA * sq_dist)
    
    # Weights are params
    # rbf_values shape: (N, 10), params shape: (10,)
    # Prediction: dot product
    pred = rbf_values @ params
    
    return pred
