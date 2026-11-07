"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid model: polynomial trend + two Gaussian bumps with adaptive widths.
    
    params[0:6]  -> polynomial coefficients for (x/3)^k, k=0..5
    params[6:8]  -> Gaussian widths (log-parameterized for BFGS stability)
    params[8:10] -> Gaussian amplitudes
    
    Fixed Gaussian centers at x = -1.5 and x = 1.5.
    """
    # Scale x to [-1, 1] for polynomial conditioning
    xs = x / 3.0
    
    # Polynomial basis (6 terms)
    result = np.zeros_like(x, dtype=np.float64)
    xs_power = np.ones_like(x, dtype=np.float64)
    for k in range(6):
        result += params[k] * xs_power
        xs_power *= xs
    
    # Two Gaussian bumps with adaptive widths
    c1, c2 = -1.5, 1.5
    
    # Log-parameterized widths: w = exp(log_w), clamped to [0.05, 5.0]
    log_w1 = params[6]
    log_w2 = params[7]
    w1 = np.clip(np.exp(log_w1), 0.05, 5.0)
    w2 = np.clip(np.exp(log_w2), 0.05, 5.0)
    
    a1 = params[8]
    a2 = params[9]
    
    result += a1 * np.exp(-((x - c1) / w1) ** 2)
    result += a2 * np.exp(-((x - c2) / w2) ** 2)
    
    return result
# EVOLVE-BLOCK-END
