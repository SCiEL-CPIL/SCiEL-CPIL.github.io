"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Approximate the hidden function using a sum of 5 Gaussian basis functions.
    The centers are fixed to spread across the input domain [-2, 2].
    The parameters are weights and widths (squared to ensure positivity) for each Gaussian.
    params: array of length 10 [w0, s0, w1, s1, w2, s2, w3, s3, w4, s4]
    """
    # Fixed centers
    centers = np.array([-1.6, -0.8, 0.0, 0.8, 1.6])
    
    y = np.zeros_like(x, dtype=float)
    
    for i in range(5):
        w = params[2*i]
        # Use squared parameter to ensure positive width, add small epsilon for stability
        sigma = params[2*i + 1]**2 + 1e-3
        y += w * np.exp(-((x - centers[i])**2) / (2.0 * sigma))
        
    return y
# EVOLVE-BLOCK-END
