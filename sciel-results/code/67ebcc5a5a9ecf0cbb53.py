"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers the target function using a mixed polynomial + sinusoidal basis in 2D.
    Uses 10 parameters: 5 for polynomial terms (1, x, y, x^2, y^2) and 5 for
    sinusoidal terms with adaptive frequencies.
    """
    p = params
    
    # Polynomial part: 1, x, y, x^2, y^2
    x2 = x * x
    y2 = y * y
    
    # Sinusoidal part with adaptive frequencies
    # p[5]: amplitude for sin(freq_x * x + freq_y * y)
    # p[6]: frequency_x
    # p[7]: frequency_y
    # p[8]: amplitude for cos(freq_x2 * x - freq_y2 * y)
    # p[9]: frequency_x2 (we'll use p[9] for another frequency or phase)
    
    # Use adaptive frequencies from params
    freq_x = p[6]
    freq_y = p[7]
    freq_x2 = p[9]
    
    # Clamp frequencies to avoid extreme values that could cause numerical issues
    freq_x = np.clip(freq_x, -20, 20)
    freq_y = np.clip(freq_y, -20, 20)
    freq_x2 = np.clip(freq_x2, -20, 20)
    
    # Sinusoidal terms
    phase1 = freq_x * x + freq_y * y
    phase2 = freq_x2 * x - freq_y * y
    
    sin1 = np.sin(phase1)
    cos2 = np.cos(phase2)
    
    # Construct the prediction
    pred = (p[0] * np.ones_like(x) +
            p[1] * x +
            p[2] * y +
            p[3] * x2 +
            p[4] * y2 +
            p[5] * sin1 +
            p[8] * cos2)
    
    # Ensure finite values
    pred = np.where(np.isfinite(pred), pred, 0.0)
    
    return pred
# EVOLVE-BLOCK-END
