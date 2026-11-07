"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Hybrid basis: polynomial + trig + exp terms with 10 params."""
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Polynomial terms
    x2 = x * x
    y2 = y * y
    z2 = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    
    # Trigonometric terms
    sin_x = np.sin(x)
    cos_x = np.cos(x)
    sin_y = np.sin(y)
    cos_y = np.cos(y)
    sin_z = np.sin(z)
    cos_z = np.cos(z)
    
    # Exponential terms (scaled to avoid overflow in [0.5, 2])
    exp_neg_x = np.exp(-x)
    exp_neg_y = np.exp(-y)
    exp_neg_z = np.exp(-z)
    
    # Log terms
    log_x = np.log(x)
    log_y = np.log(y)
    log_z = np.log(z)
    
    # Combined features
    f1 = p0 * 1.0
    f2 = p1 * (x + y + z)
    f3 = p2 * (x2 + y2 + z2)
    f4 = p3 * (xy + xz + yz)
    f5 = p4 * (sin_x + sin_y + sin_z)
    f6 = p5 * (cos_x + cos_y + cos_z)
    f7 = p6 * (exp_neg_x + exp_neg_y + exp_neg_z)
    f8 = p7 * (log_x + log_y + log_z)
    f9 = p8 * (x * sin_y + y * sin_z + z * sin_x)
    f10 = p9 * (x * exp_neg_y + y * exp_neg_z + z * exp_neg_x)
    
    return f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8 + f9 + f10
# EVOLVE-BLOCK-END
