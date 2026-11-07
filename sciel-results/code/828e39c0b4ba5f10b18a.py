# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Gaussian Radial Basis Function (RBF) network for nonlinear oscillator dynamics.
    
    The equation computes the acceleration (or dynamics variable y) as a function of time t,
    position x, and velocity v, with 10 fitted parameters.
    
    Strategy:
    - Use 10 RBF centers spread across the input domain.
    - Each RBF has a Gaussian kernel with width scaled to the domain size.
    - The 10 parameters are the weights for each RBF.
    
    Input ranges:
        t in [30, 50]  -> center spread over this range
        x in [-0.2056, 0.1749]
        v in [-0.3391, 0.3624]
    
    The RBF network provides a smooth, localized nonlinear basis that avoids the
    global ill-conditioning of high-order polynomials while capturing complex dynamics.
    """
    # Input ranges
    t_min, t_max = 30.0, 50.0
    x_min, x_max = -0.2056, 0.1749
    v_min, v_max = -0.3391, 0.3624
    
    # Define 10 RBF centers spread across the 3D input space
    # Use a stratified grid: 10 points distributed in 3D
    # We'll use a simple deterministic set of centers
    
    # Create centers using a Latin hypercube-like distribution
    n_centers = 10
    centers = np.array([
        [35.0,  0.0,   0.0  ],
        [40.0, -0.1,  -0.1  ],
        [45.0,  0.1,   0.1  ],
        [32.0, -0.15,  0.15 ],
        [48.0,  0.15, -0.15 ],
        [37.0,  0.05, -0.05 ],
        [43.0, -0.05,  0.05 ],
        [34.0,  0.12,  0.12 ],
        [46.0, -0.12, -0.12 ],
        [40.0,  0.0,   0.0  ],
    ])
    
    # Widths: scale to domain size
    # t range = 20, x range = 0.3805, v range = 0.7015
    # Use width proportional to range / sqrt(n_centers) for good coverage
    t_width = (t_max - t_min) / np.sqrt(n_centers)   # ~6.32
    x_width = (x_max - x_min) / np.sqrt(n_centers)   # ~0.1205
    v_width = (v_max - v_min) / np.sqrt(n_centers)   # ~0.2225
    
    # Compute squared distances from each data point to each center
    # t, x, v are 1D arrays of same length
    dt = t[:, np.newaxis] - centers[:, 0]  # (N, 10)
    dx = x[:, np.newaxis] - centers[:, 1]
    dv = v[:, np.newaxis] - centers[:, 2]
    
    # Squared distance in scaled space
    dist2 = (dt / t_width) ** 2 + (dx / x_width) ** 2 + (dv / v_width) ** 2
    
    # RBF values: exp(-0.5 * dist2)
    rbf_values = np.exp(-0.5 * dist2)  # (N, 10)
    
    # Combine with weights
    result = rbf_values @ params  # (N,)
    
    # Safety check for non-finite values
    if not np.all(np.isfinite(result)):
        return np.zeros_like(t)
        
    return result
# EVOLVE-BLOCK-END
