"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a nonlinear oscillator: physical prior + RBF corrections.
    params[0] = damping coefficient (linear in v)
    params[1] = linear restoring (in x)
    params[2] = cubic restoring (in x^3)
    params[3..9] = RBF amplitudes for 7 radial basis functions
    """
    # Physical prior: damped Duffing-like oscillator
    # a = -c*v - k*x - k3*x^3 + corrections
    prior = -params[0] * v - params[1] * x - params[2] * x**3

    # RBF corrections: 7 Gaussian kernels on (x, v) space
    # Centers spread across training range, widths tuned for local adaptation
    # Using tanh-mapped centers to keep them in data range during optimization
    cx_list = np.array([-0.3, 0.0, 0.3, -0.15, 0.15, -0.05, 0.05])
    cv_list = np.array([-0.1, 0.0, 0.1, 0.0, 0.0, -0.05, 0.05])
    sigma_x = 0.25
    sigma_v = 0.12

    # Compute RBF terms
    dx = x[:, np.newaxis] - cx_list[np.newaxis, :]
    dv = v[:, np.newaxis] - cv_list[np.newaxis, :]
    r2 = (dx / sigma_x)**2 + (dv / sigma_v)**2
    rbf = np.exp(-r2)  # shape: (n_samples, 7)

    # Combine: prior + RBF corrections
    correction = rbf @ params[3:10]

    return prior + correction
# EVOLVE-BLOCK-END
