"""RBF-based symbolic regression for nonlinear oscillator acceleration."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a damped nonlinear oscillator using a Gaussian RBF basis.
    
    Inputs:
        t: time, range [30, 50]
        x: position, range [-0.2056, 0.1749]
        v: velocity, range [-0.3391, 0.3624]
        params: array of 10 fitted constants
    
    The RBF network uses:
    - 4 centers in the 3D input space (t, x, v) for better coverage
    - 4 RBF weights
    - 3 linear terms (t, x, v)
    - 2 quadratic terms (x*v, v^2)
    - 1 bias term
    Total: 4 + 3 + 2 + 1 = 10 parameters
    """
    # Ensure inputs are arrays
    t = np.asarray(t, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    
    # Input ranges
    t_min, t_max = 30.0, 50.0
    x_min, x_max = -0.2056, 0.1749
    v_min, v_max = -0.3391, 0.3624
    
    # Domain sizes
    t_range = t_max - t_min  # 20
    x_range = x_max - x_min  # ~0.38
    v_range = v_max - v_min  # ~0.70
    
    # Widths scaled to domain size - use smaller widths for more localized basis
    width_scale_t = t_range / 4.0  # ~5.0
    width_scale_x = x_range / 4.0  # ~0.095
    width_scale_v = v_range / 4.0  # ~0.175
    
    # Define 4 RBF centers spread across the input space
    # Center 1: early time, negative x, negative v
    t_c1, x_c1, v_c1 = 33.0, -0.10, -0.20
    # Center 2: mid-early time, near origin
    t_c2, x_c2, v_c2 = 38.0, 0.0, 0.0
    # Center 3: mid-late time, positive x, positive v
    t_c3, x_c3, v_c3 = 43.0, 0.08, 0.15
    # Center 4: late time, extreme values
    t_c4, x_c4, v_c4 = 47.0, -0.15, 0.25
    
    # Compute squared distances for each RBF center
    # RBF 1
    dt1 = t - t_c1
    dx1 = x - x_c1
    dv1 = v - v_c1
    dist1_sq = (dt1 / width_scale_t) ** 2 + (dx1 / width_scale_x) ** 2 + (dv1 / width_scale_v) ** 2
    
    # RBF 2
    dt2 = t - t_c2
    dx2 = x - x_c2
    dv2 = v - v_c2
    dist2_sq = (dt2 / width_scale_t) ** 2 + (dx2 / width_scale_x) ** 2 + (dv2 / width_scale_v) ** 2
    
    # RBF 3
    dt3 = t - t_c3
    dx3 = x - x_c3
    dv3 = v - v_c3
    dist3_sq = (dt3 / width_scale_t) ** 2 + (dx3 / width_scale_x) ** 2 + (dv3 / width_scale_v) ** 2
    
    # RBF 4
    dt4 = t - t_c4
    dx4 = x - x_c4
    dv4 = v - v_c4
    dist4_sq = (dt4 / width_scale_t) ** 2 + (dx4 / width_scale_x) ** 2 + (dv4 / width_scale_v) ** 2
    
    # Gaussian RBFs
    rbf1 = np.exp(-0.5 * dist1_sq)
    rbf2 = np.exp(-0.5 * dist2_sq)
    rbf3 = np.exp(-0.5 * dist3_sq)
    rbf4 = np.exp(-0.5 * dist4_sq)
    
    # Linear terms
    lin_t = t
    lin_x = x
    lin_v = v
    
    # Quadratic terms for nonlinear coupling
    quad_xv = x * v
    quad_v2 = v ** 2
    
    # Parameter allocation:
    # params[0:4] -> RBF weights
    # params[4:7] -> linear weights for t, x, v
    # params[7:9] -> quadratic weights for xv, v^2
    # params[9]  -> bias
    
    # Final prediction
    pred = (params[0] * rbf1 +
            params[1] * rbf2 +
            params[2] * rbf3 +
            params[3] * rbf4 +
            params[4] * lin_t +
            params[5] * lin_x +
            params[6] * lin_v +
            params[7] * quad_xv +
            params[8] * quad_v2 +
            params[9] * 1.0)
    
    return pred
# EVOLVE-BLOCK-END
