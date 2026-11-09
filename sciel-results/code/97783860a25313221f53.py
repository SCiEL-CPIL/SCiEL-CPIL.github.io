# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid basis: 5 Chebyshev polynomials (degrees 0-4) for global trend +
    5 Gaussian RBFs with varying bandwidths for local detail.
    Linear in parameters, well-conditioned for BFGS.
    """
    x = np.asarray(x, dtype=np.float64)
    
    if not np.all(np.isfinite(x)):
        return np.zeros_like(x)
    
    p = np.asarray(params, dtype=np.float64)
    
    # Clip x to safe domain for Chebyshev mapping
    x_min, x_max = 0.1, 6.0
    x_safe = np.clip(x, x_min, x_max)
    
    # Map x from [x_min, x_max] to [-1, 1] for Chebyshev polynomials
    t = (2.0 * x_safe - (x_max + x_min)) / (x_max - x_min)
    
    # Chebyshev polynomials T_0 through T_4
    # T_0 = 1
    # T_1 = t
    # T_2 = 2*t^2 - 1
    # T_3 = 4*t^3 - 3*t
    # T_4 = 8*t^4 - 8*t^2 + 1
    T0 = np.ones_like(t)
    T1 = t
    T2 = 2.0 * t * t - 1.0
    T3 = 4.0 * t**3 - 3.0 * t
    T4 = 8.0 * t**4 - 8.0 * t * t + 1.0
    
    # 5 Gaussian RBFs with fixed centers and varying bandwidths
    # Centers evenly spaced across [0.1, 6]
    centers = np.array([0.7, 1.9, 3.05, 4.2, 5.4])
    
    # Varying bandwidths: wider in the middle, narrower at edges
    # This allows better local detail where the function might be more complex
    sigmas = np.array([0.3, 0.4, 0.5, 0.4, 0.3])
    
    pred = np.zeros_like(x)
    
    # Add Chebyshev components
    pred += p[0] * T0
    pred += p[1] * T1
    pred += p[2] * T2
    pred += p[3] * T3
    pred += p[4] * T4
    
    # Add RBF components
    for i in range(5):
        diff = x - centers[i]
        rbf = np.exp(-(diff * diff) / (2.0 * sigmas[i] * sigmas[i]))
        pred += p[5 + i] * rbf
    
    # Ensure finite output
    pred = np.where(np.isfinite(pred), pred, 0.0)
    
    return pred
# EVOLVE-BLOCK-END
