# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression with a hybrid basis:
    - Low-order polynomial (constant, linear, quadratic)
    - Learnable sigmoid (amplitude, center, steepness)
    - Two Gaussian RBFs with learnable centers and fixed widths
    
    Parameters (10 total):
    0: Constant
    1: Linear x
    2: Quadratic x^2
    3: Sigmoid amplitude
    4: Sigmoid center
    5: Sigmoid steepness
    6: Gaussian 1 amplitude
    7: Gaussian 1 center
    8: Gaussian 2 amplitude
    9: Gaussian 2 center
    
    The sigmoid captures step-like/saturating behaviors.
    The Gaussians capture local bumps/peaks.
    The polynomial captures global trend.
    """
    x = np.asarray(x, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    
    # Ensure p has exactly 10 elements
    if p.size < 10:
        p = np.concatenate([p, np.zeros(10 - p.size)])
    p = p[:10]
    
    p0, p1, p2 = p[0], p[1], p[2]
    A_sig, c_sig, k_sig = p[3], p[4], p[5]
    A_g1, c_g1 = p[6], p[7]
    A_g2, c_g2 = p[8], p[9]
    
    # Polynomial part
    poly = p0 + p1 * x + p2 * x * x
    
    # Sigmoid part: A / (1 + exp(-k * (x - c)))
    z = k_sig * (x - c_sig)
    z_clipped = np.clip(z, -50, 50)
    sigmoid = A_sig / (1.0 + np.exp(-z_clipped))
    
    # Gaussian RBFs with fixed width 0.4
    width = 0.4
    g1 = A_g1 * np.exp(-((x - c_g1) ** 2) / (2.0 * width ** 2))
    g2 = A_g2 * np.exp(-((x - c_g2) ** 2) / (2.0 * width ** 2))
    
    result = poly + sigmoid + g1 + g2
    
    return result
# EVOLVE-BLOCK-END
