import numpy as np
import time

# Normalization constants based on training range
# x in [-0.5728, 0.5275], v in [-0.1754, 0.2821]
x_min, x_max = -0.5728, 0.5275
v_min, v_max = -0.1754, 0.2821

x_center = (x_max + x_min) / 2.0
x_scale = (x_max - x_min) / 2.0
v_center = (v_max + v_min) / 2.0
v_scale = (v_max - v_min) / 2.0

# RBF centers in normalized space
# We use 10 parameters as 10 RBF weights
# Centers are fixed at well-distributed points in the normalized domain
# This is a structural change from polynomial basis to kernel-based local approximation

# Define 10 centers in the 2D normalized space (xn, vn) in [-1, 1] x [-1, 1]
# Use a grid-like distribution with some jitter for better coverage
centers_x = np.array([-0.8, -0.4, 0.0, 0.4, 0.8, -0.6, 0.2, -0.2, 0.6, 0.0])
centers_v = np.array([-0.6, 0.2, -0.2, 0.4, -0.4, 0.0, 0.6, -0.6, 0.0, 0.2])

# RBF bandwidth - fixed, not a parameter (to save parameter budget)
# A reasonable bandwidth for normalized inputs in [-1, 1]
sigma = 0.4

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    RBF (Radial Basis Function) kernel regression model.
    
    10 parameters are the weights for 10 Gaussian kernels centered at
    fixed locations in the normalized input space.
    
    Inputs:
    x: position
    v: velocity
    params: 10 RBF weights
    
    Output:
    Prediction array with one value per row.
    """
    # Normalize inputs
    xn = (x - x_center) / x_scale
    vn = (v - v_center) / v_scale
    
    # Compute RBF values for each center
    # params[i] * exp(-((xn - cx_i)^2 + (vn - cv_i)^2) / (2 * sigma^2))
    
    result = np.zeros_like(xn, dtype=np.float64)
    
    for i in range(10):
        dx = xn - centers_x[i]
        dv = vn - centers_v[i]
        dist_sq = dx * dx + dv * dv
        rbf_val = np.exp(-dist_sq / (2.0 * sigma * sigma))
        result += params[i] * rbf_val
    
    # Ensure finite values
    result = np.where(np.isfinite(result), result, 0.0)
    
    return result
