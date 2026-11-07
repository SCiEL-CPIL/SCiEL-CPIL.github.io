"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    RBF (Gaussian kernel) basis expansion for nonlinear oscillator acceleration.
    
    Uses 10 parameters to define a sum of 3 Gaussian bumps in (t, x, v) state space:
    - p[0:3]:  weights (amplitudes) for 3 bumps
    - p[3:6]:  center offsets added to fixed centers [-1, 0, 1]
    - p[6:9]:  shared scales (sigma_t, sigma_x, sigma_v)
    - p[9]:    global offset
    
    Fixed centers at [-1, 0, 1] in each dimension ensure the bumps span the
    training range and BFGS starting from all-ones has a well-conditioned landscape.
    """
    # Extract parameters
    w = params[0:3]       # weights
    c_off = params[3:6]   # center offsets
    scales = params[6:9]  # shared scales
    offset = params[9]    # global offset

    # Fixed base centers in each dimension
    base = np.array([-1.0, 0.0, 1.0])
    centers = base[None, :] + c_off[:, None]  # shape (3, 3): each row = [t_c, x_c, v_c]

    # Ensure scales are positive and not too small
    sig_t = np.clip(scales[0], 0.05, 10.0)
    sig_x = np.clip(scales[1], 0.05, 10.0)
    sig_v = np.clip(scales[2], 0.05, 10.0)

    # Compute squared distances from each data point to each bump center
    # t, x, v are 1D arrays of shape (N,)
    # centers is (3, 3)
    dt = (t[None, :] - centers[:, 0]) / sig_t  # shape (3, N)
    dx = (x[None, :] - centers[:, 1]) / sig_x  # shape (3, N)
    dv = (v[None, :] - centers[:, 2]) / sig_v  # shape (3, N)

    d2 = dt * dt + dx * dx + dv * dv  # shape (3, N)

    # Gaussian kernels: exp(-d2), clamped to avoid underflow issues
    kernels = np.exp(-np.clip(d2, 0.0, 50.0))  # shape (3, N)

    # Weighted sum of kernels + offset
    result = w[None, :] @ kernels + offset  # shape (N,)

    return result
# EVOLVE-BLOCK-END
