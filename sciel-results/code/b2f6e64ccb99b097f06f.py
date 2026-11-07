"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Design: Piecewise-linear RBFs (L1 norm based) with 9 centers + 1 bias.
    # Centers: 8 corners of the domain [0.5, 2.0]^3 and the center (1.25, 1.25, 1.25).
    # Kernel: max(0, 1 - L1_distance / R)
    # R is chosen to ensure overlap between corner and center RBFs.
    # Max L1 distance from corner to center is 2.25.
    # Let R = 1.5. This ensures coverage.
    
    centers = np.array([
        [0.5, 0.5, 0.5],
        [2.0, 0.5, 0.5],
        [0.5, 2.0, 0.5],
        [2.0, 2.0, 0.5],
        [0.5, 0.5, 2.0],
        [2.0, 0.5, 2.0],
        [0.5, 2.0, 2.0],
        [2.0, 2.0, 2.0],
        [1.25, 1.25, 1.25]
    ])
    
    R = 1.5
    
    # x, y, z are arrays of shape (N,)
    points = np.stack([x, y, z], axis=1)  # (N, 3)
    
    # Compute L1 distances to each center
    # diff shape (N, 9, 3)
    diff = points[:, np.newaxis, :] - centers[np.newaxis, :, :]
    l1_dist = np.sum(np.abs(diff), axis=2)  # (N, 9)
    
    # Apply piecewise-linear kernel
    # K = max(0, 1 - d/R)
    rbf_vals = np.maximum(0.0, 1.0 - l1_dist / R)  # (N, 9)
    
    # Parameters:
    # params[0:9] for RBFs
    # params[9] for bias
    
    p_rbf = params[0:9]
    p_bias = params[9]
    
    # Combine
    pred_rbf = rbf_vals @ p_rbf
    pred_bias = p_bias
    
    return pred_rbf + pred_bias
# EVOLVE-BLOCK-END
