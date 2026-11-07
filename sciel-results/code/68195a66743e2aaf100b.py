import numpy as np
import time

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Gaussian Radial Basis Function (RBF) network for nonlinear oscillator dynamics.
    Uses 10 RBF centers in the 3D input space (t, x, v) with fixed widths.
    Each param scales one RBF center's contribution.
    """
    # Input ranges
    t_min, t_max = 30.0, 50.0
    x_min, x_max = -0.2056, 0.1749
    v_min, v_max = -0.3391, 0.3624
    
    # Center locations: spread across the domain
    # We use 10 centers, each with distinct (t, x, v) coordinates
    # t centers: spread across [30, 50]
    # x centers: spread across [-0.2056, 0.1749]
    # v centers: spread across [-0.3391, 0.3624]
    
    # Define 10 centers manually for good coverage
    centers = np.array([
        [32.0, -0.15, -0.25],
        [36.0, 0.0, 0.0],
        [40.0, 0.10, 0.20],
        [44.0, -0.10, 0.15],
        [48.0, 0.05, -0.20],
        [34.0, 0.12, -0.10],
        [38.0, -0.18, 0.30],
        [42.0, 0.08, -0.30],
        [46.0, -0.05, 0.10],
        [50.0, 0.15, 0.05],
    ])
    
    # Widths: scale to domain size
    # t range = 20, x range = 0.3805, v range = 0.7015
    # Use widths that are ~1/5 of domain range for good localization
    t_width = 4.0  # 20/5
    x_width = 0.08  # 0.3805/5
    v_width = 0.14  # 0.7015/5
    
    # Compute RBF values
    # For each center, compute the Gaussian distance
    n_centers = 10
    
    # Precompute squared distances
    dt = t[:, np.newaxis] - centers[:, 0]  # (N, 10)
    dx = x[:, np.newaxis] - centers[:, 1]  # (N, 10)
    dv = v[:, np.newaxis] - centers[:, 2]  # (N, 10)
    
    # Squared distances normalized by widths
    dist_sq = (dt / t_width) ** 2 + (dx / x_width) ** 2 + (dv / v_width) ** 2
    
    # Gaussian RBF: exp(-0.5 * dist_sq)
    rbf_values = np.exp(-0.5 * dist_sq)  # (N, 10)
    
    # Combine with params
    # params shape is (10,)
    return rbf_values @ params
