"""RBF-based symbolic regression with 1 input."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = 145.0  # 5s safety margin before 150s kill


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Deadline guard: return zeros if we're running out of time
    if time.time() > _DEADLINE:
        x_arr = np.asarray(x, dtype=float)
        if x_arr.ndim > 1:
            x_arr = x_arr.ravel()
        return np.zeros_like(x_arr, dtype=float)

    x = np.asarray(x, dtype=float)
    if x.ndim > 1:
        x = x.ravel()
    
    p = np.asarray(params, dtype=float)
    
    # Ensure finite params
    p = np.where(np.isfinite(p), p, 1.0)
    
    # Clip extreme parameter values for numerical stability
    p = np.clip(p, -1e4, 1e4)
    
    # RBF network with 5 fixed, equally spaced centers
    # Centers at -3, -1.5, 0, 1.5, 3
    centers = np.array([-3.0, -1.5, 0.0, 1.5, 3.0])
    
    # Width scaling: p[9] controls the width, default to 1.0
    # Use a reasonable range for width scaling
    width_scale = np.clip(p[9], 0.1, 5.0)
    sigma = 1.0 / width_scale
    
    # Compute RBF features: exp(-sigma^2 * (x - center)^2)
    # Shape: (n_points, 5)
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    rbf_features = np.exp(-sigma * sigma * diff * diff)
    
    # The 5 kernel weights are p[0:5]
    kernel_weights = p[0:5]
    
    # Global offset is p[5]
    offset = p[5]
    
    # Additional basis functions for flexibility: p[6], p[7], p[8]
    # p[6] * x, p[7] * x^2, p[8] * sin(x)
    extra_basis = np.column_stack([
        x,
        x * x,
        np.sin(x)
    ])
    
    result = rbf_features @ kernel_weights + extra_basis @ p[6:9] + offset
    
    # Ensure finite output
    result = np.where(np.isfinite(result), result, 0.0)
    
    return result
# EVOLVE-BLOCK-END
