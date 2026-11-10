"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid symbolic regression model with 10 parameters.
    
    Combines a stable linear basis (polynomials, trig, tanh, Gaussian) for global
    trend fitting with a non-linear structural term (centered Gaussian) for local
    features. The linear part is convex in params, allowing BFGS to reliably
    minimize the global component. The non-linear part captures localized bumps
    or dips that a pure linear basis might miss.
    
    Parameters:
    p[0..7]: Coefficients for the linear basis:
             1, x, x^2, x^3, x^4, sin(x), cos(x), tanh(x)
    p[8]:    Amplitude of the centered Gaussian
    p[9]:    Center of the centered Gaussian (width fixed at 1.0 for stability)
    
    The Gaussian term is: p[8] * exp(-(x - p[9])^2)
    """
    # Deadline guard
    start_time = time.time()
    
    x = np.asarray(x, dtype=np.float64)
    p = params
    
    # Pre-compute basis functions
    x2 = x * x
    x3 = x2 * x
    x4 = x2 * x2
    
    sin_x = np.sin(x)
    cos_x = np.cos(x)
    tanh_x = np.tanh(x)
    
    # Linear basis (8 terms)
    b0 = np.ones_like(x)
    b1 = x
    b2 = x2
    b3 = x3
    b4 = x4
    b5 = sin_x
    b6 = cos_x
    b7 = tanh_x
    
    # Non-linear structural term (2 params: amplitude, center)
    # Fixed width = 1.0 to avoid overflow/underflow issues with BFGS
    # and to keep the landscape smooth.
    amp = p[8]
    center = p[9]
    
    # Clip center to a reasonable range to prevent extreme values
    # Inputs are in [-2, 2], so center outside [-5, 5] is unlikely to help
    center_clipped = np.clip(center, -5.0, 5.0)
    
    gauss_term = amp * np.exp(-(x - center_clipped) ** 2)
    
    # Combine all terms
    result = (
        p[0] * b0 +
        p[1] * b1 +
        p[2] * b2 +
        p[3] * b3 +
        p[4] * b4 +
        p[5] * b5 +
        p[6] * b6 +
        p[7] * b7 +
        gauss_term
    )
    
    # Ensure finite output
    result = np.nan_to_num(result, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # Check deadline
    elapsed = time.time() - start_time
    if elapsed > 110.0:
        return np.zeros_like(x)
        
    return result
# EVOLVE-BLOCK-END
