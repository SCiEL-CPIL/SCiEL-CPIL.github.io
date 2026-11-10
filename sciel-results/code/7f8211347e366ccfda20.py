"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Sum of 5 Gaussian radial basis functions with non-linear parameterization.
    
    Parameters (10 total):
      p[0:5]  - amplitudes of the 5 Gaussians
      p[5:10] - means (centers) of the 5 Gaussians
    
    The widths are fixed at 0.8 to keep the problem well-conditioned for BFGS.
    """
    x = np.asarray(x, dtype=np.float64)
    p = params
    
    # Extract parameters
    amplitudes = p[0:5]
    centers = p[5:10]
    
    # Fixed width for stability
    width = 0.8
    
    # Compute sum of Gaussians
    y = np.zeros_like(x)
    for amp, center in zip(amplitudes, centers):
        y += amp * np.exp(-((x - center) ** 2) / (2 * width ** 2))
    
    return y
# EVOLVE-BLOCK-END
