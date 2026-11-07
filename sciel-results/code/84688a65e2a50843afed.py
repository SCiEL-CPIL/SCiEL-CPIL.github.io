# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Adaptive symbolic regression model with 10 parameters.
    
    Structure:
    f(x) = p0 + p1*x + p2*x^2 + p3*x^3
         + p4*sin(p5*x) + p6*cos(p7*x)
         + p8*tanh(p9*x)
    
    This uses a mix of polynomial, Fourier, and saturating basis functions.
    All parameters appear smoothly, allowing BFGS to optimize from all-ones start.
    """
    p = np.asarray(params, dtype=float)
    x = np.asarray(x, dtype=float)
    
    # Ensure finite parameters
    if not np.all(np.isfinite(p)):
        return np.full_like(x, np.nan)
    
    # Polynomial part
    poly = p[0] + p[1]*x + p[2]*x*x + p[3]*x*x*x
    
    # Fourier part with parametric frequencies
    # Clamp frequencies to avoid numerical issues
    f1 = np.clip(p[5], -50, 50)
    f2 = np.clip(p[7], -50, 50)
    
    fourier = p[4]*np.sin(f1*x) + p[6]*np.cos(f2*x)
    
    # Tanh saturation with parametric scale
    scale = np.clip(p[9], -50, 50)
    tanh_term = p[8]*np.tanh(scale*x)
    
    y = poly + fourier + tanh_term
    
    # Ensure output is finite
    y = np.where(np.isfinite(y), y, 0.0)
    
    return y
# EVOLVE-BLOCK-END
