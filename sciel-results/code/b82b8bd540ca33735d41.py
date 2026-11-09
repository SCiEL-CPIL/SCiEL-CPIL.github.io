"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Structural change: Non-linear parametric model.
    Uses a sum of Gaussian, Sine, Polynomial, and Tanh terms.
    Parameters control shape (width, center, freq, phase) not just amplitude.
    """
    p = params
    
    # Unpack parameters
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = p
    
    x_safe = np.clip(x, -10, 10)
    
    # 1. Gaussian Term: p0 * exp(-exp(p1) * (x - p2)^2)
    # Clip p1 to avoid overflow in exp
    p1_safe = np.clip(p1, -20, 20)
    width = np.exp(p1_safe)
    gauss = p0 * np.exp(-width * (x_safe - p2) ** 2)
    
    # 2. Sine Term: p3 * sin(p4 * x + p5)
    sine = p3 * np.sin(p4 * x_safe + p5)
    
    # 3. Polynomial Term: p6 * x + p7 * x^2
    poly = p6 * x_safe + p7 * x_safe ** 2
    
    # 4. Tanh Term: p8 * tanh(p9 * x)
    tanh_term = p8 * np.tanh(p9 * x_safe)
    
    result = gauss + sine + poly + tanh_term
    
    # Ensure finite output
    result = np.nan_to_num(result, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return result
# EVOLVE-BLOCK-END
