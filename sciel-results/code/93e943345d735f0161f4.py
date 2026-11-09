"""Symbolic regression with nonlinear parametric model."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Nonlinear parametric model with 10 parameters controlling:
    - Two Gaussian bumps: center, width, amplitude (6 params)
    - Polynomial trend: constant, linear, quadratic (3 params)
    - Global scale (1 param)
    
    This creates a rich nonlinear landscape for BFGS to navigate.
    """
    x = np.asarray(x, dtype=np.float64)
    x = np.clip(x, -10, 10)  # safety
    
    # Extract parameters with safe defaults
    # Params: [amp1, c1, w1, amp2, c2, w2, p0, p1, p2, scale]
    amp1 = params[0]
    c1 = params[1]
    w1 = np.abs(params[2]) + 0.1  # width must be positive
    amp2 = params[3]
    c2 = params[4]
    w2 = np.abs(params[5]) + 0.1
    p0 = params[6]
    p1 = params[7]
    p2 = params[8]
    scale = params[9]
    
    # Two Gaussian bumps
    g1 = amp1 * np.exp(-((x - c1) ** 2) / (2.0 * w1 ** 2))
    g2 = amp2 * np.exp(-((x - c2) ** 2) / (2.0 * w2 ** 2))
    
    # Polynomial trend
    poly = p0 + p1 * x + p2 * x ** 2
    
    # Combine with global scale
    y = scale * (g1 + g2 + poly)
    
    # Ensure finite output
    y = np.nan_to_num(y, nan=0.0, posinf=1e10, neginf=-1e10)
    
    return y
# EVOLVE-BLOCK-END
