"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Composite model: 5 Gaussian bumps with adaptive centers and widths
    # params[0], params[1]: amplitude, center for bump 1
    # params[2], params[3]: amplitude, center for bump 2
    # params[4], params[5]: amplitude, center for bump 3
    # params[6], params[7]: amplitude, center for bump 4
    # params[8], params[9]: amplitude, center for bump 5
    # Width is fixed at 0.5 for stability
    
    x = np.asarray(x, dtype=float)
    result = np.zeros_like(x)
    
    for i in range(5):
        amp = params[2 * i]
        center = params[2 * i + 1]
        width = 0.5
        result += amp * np.exp(-((x - center) ** 2) / (2 * width ** 2))
    
    return result
# EVOLVE-BLOCK-END
