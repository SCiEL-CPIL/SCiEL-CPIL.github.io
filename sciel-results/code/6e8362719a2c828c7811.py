"""Program for symbolic regression task using scaled polynomial + trigonometric basis."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Model combining scaled polynomial and trigonometric terms.
    
    The input x is scaled to [-1, 1] by dividing by 3.0.
    
    params[0:6]  -> coefficients for (x/3)^k, k=0..5 (scaled polynomial)
    params[6:8]  -> amplitudes for sin(pi*x/3) and cos(pi*x/3)
    params[8:10] -> amplitudes for sin(2*pi*x/3) and cos(2*pi*x/3)
    
    Args:
        x: Input array of shape (N,).
        params: Array of 10 parameters.
    
    Returns:
        Prediction array of shape (N,).
    """
    x = np.asarray(x, dtype=float)
    p = np.asarray(params, dtype=float)
    
    # Safety check for non-finite params
    if not np.all(np.isfinite(p)):
        return np.zeros_like(x)
    
    # Scale input to [-1, 1] for better-conditioned polynomial
    xs = x / 3.0
    
    # Scaled polynomial part: sum_{k=0}^{5} p[k] * xs^k
    # Horner's method for numerical stability
    poly_part = p[5]
    for k in range(4, -1, -1):
        poly_part = poly_part * xs + p[k]
    
    # Trigonometric part
    # Use scaled argument for trig to capture periodic behavior
    theta = np.pi * xs  # theta in [-pi/3, pi/3]
    trig_part = (p[6] * np.sin(theta) + p[7] * np.cos(theta) +
                 p[8] * np.sin(2.0 * theta) + p[9] * np.cos(2.0 * theta))
    
    result = poly_part + trig_part
    
    # Safety: ensure finite output
    if not np.all(np.isfinite(result)):
        return np.zeros_like(x)
    
    return result

# EVOLVE-BLOCK-END
