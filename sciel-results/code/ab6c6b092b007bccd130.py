"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # RBF network with 5 fixed centers across [-3, 3].
    # 10 params: [amp1, width1, amp2, width2, amp3, width3, amp4, width4, amp5, width5]
    # Each Gaussian: amp_i * exp(-width_i * (x - center_i)^2)
    # Widths are clamped to positive values for stability.
    
    centers = np.array([-3.0, -1.5, 0.0, 1.5, 3.0])
    
    result = np.zeros_like(x)
    for i in range(5):
        amp = params[2 * i]
        width = np.maximum(params[2 * i + 1], 1e-3)
        result += amp * np.exp(-width * (x - centers[i]) ** 2)
    
    return result
# EVOLVE-BLOCK-END
