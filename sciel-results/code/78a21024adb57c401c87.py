"""Symbolic regression using a fixed Radial Basis Function (RBF) network."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Model using a fixed RBF basis.
    Linear in parameters, so BFGS converges reliably.
    
    The basis consists of Gaussian kernels centered at -2, -1, 0, 1, 2 
    with two different widths (sigma=1 and sigma=2).
    
    params (length 10):
        p0-p4: Coefficients for narrow Gaussians (sigma=1) at centers -2, -1, 0, 1, 2
        p5-p9: Coefficients for wide Gaussians (sigma=2) at centers -2, -1, 0, 1, 2
    """
    x = np.asarray(x, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)

    # Centers
    c = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    
    # Reshape x for broadcasting: (N, 1)
    x_col = x[:, np.newaxis]
    
    # Narrow Gaussians (sigma=1, variance=1)
    # exp(-(x-c)^2)
    diff_narrow = x_col - c[np.newaxis, :]
    b_narrow = np.exp(-diff_narrow * diff_narrow)
    
    # Wide Gaussians (sigma=2, variance=4)
    # exp(-(x-c)^2 / 4)
    b_wide = np.exp(-diff_narrow * diff_narrow / 4.0)
    
    # Combine linearly
    # b_narrow is shape (N, 5), b_wide is shape (N, 5)
    # params is shape (10,)
    # We can stack them into a (N, 10) matrix and dot product
    # Or just sum manually to avoid memory overhead if N is huge, but N=200 is small.
    
    result = (
        params[0] * b_narrow[:, 0] +
        params[1] * b_narrow[:, 1] +
        params[2] * b_narrow[:, 2] +
        params[3] * b_narrow[:, 3] +
        params[4] * b_narrow[:, 4] +
        params[5] * b_wide[:, 0] +
        params[6] * b_wide[:, 1] +
        params[7] * b_wide[:, 2] +
        params[8] * b_wide[:, 3] +
        params[9] * b_wide[:, 4]
    )

    # Ensure finite output
    return np.nan_to_num(result, nan=0.0, posinf=1e8, neginf=-1e8)
# EVOLVE-BLOCK-END
