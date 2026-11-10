"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Approximate the hidden function using a Fourier series with 5 harmonics.
    This basis is effective for periodic or oscillatory functions.
    params: [a0, a1, b1, a2, b2, a3, b3, a4, b4, a5]
    """
    # Scale x to [-pi, pi]
    t = (np.pi / 2.0) * x
    
    y = np.zeros_like(x, dtype=float)
    y += params[0]  # DC term
    
    # Harmonics 1 to 4 have both sine and cosine
    # Harmonic 5 has only cosine (to fit exactly 10 params: 1 + 4*2 + 1 = 10)
    
    for k in range(1, 5):
        y += params[2 * k - 1] * np.cos(k * t)
        y += params[2 * k] * np.sin(k * t)
        
    # 5th harmonic cosine
    y += params[9] * np.cos(5 * t)
    
    return y
# EVOLVE-BLOCK-END
