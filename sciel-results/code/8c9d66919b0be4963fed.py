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
    - 3 centers in the 3D input space (t, x, v)
    - 3 widths (one per center)
    - 3 additional linear terms for baseline behavior
    
    This gives 3 + 3 + 3 = 9 parameters, plus 1 bias = 10 total.
    """
    # Ensure inputs are arrays
    t = np.asarray(t, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    
    # Input ranges
    t_min, t_max = 30.0, 50.0
    x_min, x_max = -0.2056, 0.1749
    v_min, v_max = -0.3391, 0.3624
    
    # Define RBF centers (fixed, distinct values within input ranges)
    # Center 1: lower-left region
    t_c1, x_c1, v_c1 = 35.0, -0.05, -0.15
    # Center 2: middle region
    t_c2, x_c2, v_c2 = 40.0, 0.0, 0.0
    # Center 3: upper-right region
    t_c3, x_c3, v_c3 = 45.0, 0.10, 0.20
    
    # Widths scaled to domain size (fixed fractions of domain range)
    # Domain sizes
    t_range = t_max - t_min  # 20
    x_range = x_max - x_min  # ~0.38
    v_range = v_max - v_min  # ~0.70
    
    # Widths as fractions of domain size
    width_scale_t = t_range / 3.0  # ~6.67
    width_scale_x = x_range / 3.0  # ~0.127
    width_scale_v = v_range / 3.0  # ~0.233
    
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
    
    # Gaussian RBFs
    rbf1 = np.exp(-0.5 * dist1_sq)
    rbf2 = np.exp(-0.5 * dist2_sq)
    rbf3 = np.exp(-0.5 * dist3_sq)
    
    # Linear baseline terms
    lin_t = t
    lin_x = x
    lin_v = v
    
    # Combine: 3 RBF weights + 3 linear weights + 1 bias = 7 params
    # We have 10 params total. Let's use:
    # params[0:3] -> RBF weights
    # params[3:6] -> linear weights for t, x, v
    # params[6]  -> bias
    # params[7:10] -> additional quadratic terms for richer dynamics
    
    # Quadratic terms
    quad_x = x ** 2
    quad_v = v ** 2
    quad_xv = x * v
    
    # Final prediction
    pred = (params[0] * rbf1 +
            params[1] * rbf2 +
            params[2] * rbf3 +
            params[3] * lin_t +
            params[4] * lin_x +
            params[5] * lin_v +
            params[6] * 1.0 +
            params[7] * quad_x +
            params[8] * quad_v +
            params[9] * quad_xv)
    
    return pred
# EVOLVE-BLOCK-END
