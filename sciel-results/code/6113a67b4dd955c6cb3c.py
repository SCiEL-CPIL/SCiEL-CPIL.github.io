# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Gaussian Radial Basis Function (RBF) network for nonlinear oscillator dynamics.
    
    Uses 10 RBF centers spread across the input domain. Each RBF is a Gaussian kernel
    in 3D space (t, x, v) with width scaled to the domain size. The 10 parameters are
    the weights for each RBF.
    
    Input ranges:
    - t: [30, 50]
    - x: [-0.2056, 0.1749]
    - v: [-0.3391, 0.3624]
    
    Centers are fixed (not fitted) to provide a stable, localized basis.
    """
    # Define RBF centers in 3D space (t, x, v)
    # Spread centers across the domain
    t_min, t_max = 30.0, 50.0
    x_min, x_max = -0.2056, 0.1749
    v_min, v_max = -0.3391, 0.3624
    
    # Create 10 centers distributed across the 3D domain
    # Use a structured grid approach
    centers = np.array([
        [35.0, -0.1, -0.2],
        [35.0,  0.0,  0.0],
        [35.0,  0.1,  0.2],
        [40.0, -0.1,  0.0],
        [40.0,  0.0, -0.2],
        [40.0,  0.1, -0.1],
        [45.0, -0.15, 0.1],
        [45.0,  0.05, 0.0],
        [45.0,  0.15, -0.15],
        [47.5,  0.0,  0.1]
    ])
    
    # Compute domain sizes for scaling
    t_range = t_max - t_min  # 20
    x_range = x_max - x_min  # ~0.38
    v_range = v_max - v_min  # ~0.70
    
    # Set widths (inverse of sigma squared) for each dimension
    # Use different widths to account for different scales
    sigma_t = t_range / 4.0  # ~5
    sigma_x = x_range / 2.0  # ~0.19
    sigma_v = v_range / 2.0  # ~0.35
    
    # Compute squared distances from each data point to each center
    # t, x, v are 1D arrays of same length
    n = len(t)
    n_centers = len(centers)
    
    # Initialize result
    result = np.zeros(n)
    
    # Compute each RBF contribution
    for i in range(n_centers):
        # Distance in 3D space, scaled by sigma
        dt = (t - centers[i, 0]) / sigma_t
        dx = (x - centers[i, 1]) / sigma_x
        dv = (v - centers[i, 2]) / sigma_v
        
        # Squared distance
        dist_sq = dt * dt + dx * dx + dv * dv
        
        # Gaussian RBF
        rbf = np.exp(-0.5 * dist_sq)
        
        # Add weighted contribution
        result += params[i] * rbf
    
    # Safety check for non-finite values
    if not np.all(np.isfinite(result)):
        return np.zeros_like(t)
        
    return result
# EVOLVE-BLOCK-END
