"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid basis: Chebyshev polynomials (up to degree 4) + Gaussian RBFs at multiple centers.
    This provides good conditioning for BFGS while capturing both smooth global trends
    and local features.
    """
    x = np.asarray(x, dtype=float)
    
    # Map x from [-2, 2] to [-1, 1] for Chebyshev
    z = x / 2.0
    
    # Chebyshev polynomials T0..T4
    T0 = np.ones_like(z)
    T1 = z
    T2 = 2 * z**2 - 1
    T3 = 4 * z**3 - 3 * z
    T4 = 8 * z**4 - 8 * z**2 + 1
    
    # Gaussian RBFs at centers -1.5, -0.75, 0, 0.75, 1.5 with width ~0.8
    sigma = 0.8
    g1 = np.exp(-((x - 1.5)**2) / (2 * sigma**2))
    g2 = np.exp(-((x - 0.75)**2) / (2 * sigma**2))
    g3 = np.exp(-(x**2) / (2 * sigma**2))
    g4 = np.exp(-((x + 0.75)**2) / (2 * sigma**2))
    g5 = np.exp(-((x + 1.5)**2) / (2 * sigma**2))
    
    # Combine: 5 Chebyshev + 5 Gaussians = 10 parameters
    y = (params[0] * T0 +
         params[1] * T1 +
         params[2] * T2 +
         params[3] * T3 +
         params[4] * T4 +
         params[5] * g1 +
         params[6] * g2 +
         params[7] * g3 +
         params[8] * g4 +
         params[9] * g5)
         
    return y
# EVOLVE-BLOCK-END
