# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Approximates the target function using a combination of:
    - Polynomial terms (up to degree 3)
    - Sinusoidal terms with learnable frequencies
    - Exponential terms
    This richer basis allows BFGS to fit a wider class of non-linear functions.
    """
    x = np.asarray(x, dtype=float).flatten()
    
    if not np.all(np.isfinite(x)):
        return np.zeros_like(x)
    
    n_points = len(x)
    if n_points == 0:
        return np.array([])
    
    # Extract parameters:
    # params[0:4] = polynomial coefficients (1, x, x^2, x^3)
    # params[4:6] = sin frequency, sin amplitude
    # params[6:8] = cos frequency, cos amplitude
    # params[8:10] = exp rate, exp amplitude
    
    a0, a1, a2, a3 = params[0], params[1], params[2], params[3]
    f_sin, amp_sin = params[4], params[5]
    f_cos, amp_cos = params[6], params[7]
    r_exp, amp_exp = params[8], params[9]
    
    # Polynomial part
    poly = a0 + a1 * x + a2 * x**2 + a3 * x**3
    
    # Sinusoidal part
    # Use a safe frequency to avoid overflow in sin/cos
    safe_f_sin = np.clip(f_sin, -10, 10)
    safe_f_cos = np.clip(f_cos, -10, 10)
    sin_part = amp_sin * np.sin(safe_f_sin * x)
    cos_part = amp_cos * np.cos(safe_f_cos * x)
    
    # Exponential part (clipped to avoid overflow)
    safe_r = np.clip(r_exp, -2, 2)
    exp_part = amp_exp * np.exp(safe_r * x)
    
    return poly + sin_part + cos_part + exp_part
# EVOLVE-BLOCK-END
