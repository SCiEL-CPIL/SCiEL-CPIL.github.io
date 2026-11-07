# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Trigonometric harmonic basis with non-integer frequency multipliers.
    
    The current program uses a quadratic polynomial (Legendre) basis which scores ~7.39.
    The best inspiration uses a mix of linear, sin, and exp terms scoring ~1.30.
    
    This program adopts a purely trigonometric harmonic basis with fixed, 
    non-integer frequency multipliers to capture potential oscillatory or 
    wave-like structures in the hidden function. This is structurally different 
    from polynomial and exponential bases.
    
    Basis functions (10 total):
    params[0]: 1 (constant / DC offset)
    params[1]: sin(2.0 * x)
    params[2]: sin(3.0 * y)
    params[3]: sin(4.0 * z)
    params[4]: cos(2.5 * x)
    params[5]: cos(3.5 * y)
    params[6]: cos(4.5 * z)
    params[7]: sin(x + y)
    params[8]: sin(y + z)
    params[9]: sin(z + x)
    
    All terms are linear in params, so BFGS optimization remains convex.
    """
    p = params
    
    # Precompute trigonometric components
    sin_2x = np.sin(2.0 * x)
    sin_3y = np.sin(3.0 * y)
    sin_4z = np.sin(4.0 * z)
    
    cos_2p5x = np.cos(2.5 * x)
    cos_3p5y = np.cos(3.5 * y)
    cos_4p5z = np.cos(4.5 * z)
    
    sin_xp_y = np.sin(x + y)
    sin_yp_z = np.sin(y + z)
    sin_zp_x = np.sin(z + x)
    
    return (p[0]
            + p[1] * sin_2x
            + p[2] * sin_3y
            + p[3] * sin_4z
            + p[4] * cos_2p5x
            + p[5] * cos_3p5y
            + p[6] * cos_4p5z
            + p[7] * sin_xp_y
            + p[8] * sin_yp_z
            + p[9] * sin_zp_x)
# EVOLVE-BLOCK-END
