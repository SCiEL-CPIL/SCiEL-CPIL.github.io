"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE = time.time() + 140.0  # 10s safety margin before 150s kill


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    p = params

    # 8 Gaussian basis functions centered at evenly spaced points in [-3, 3]
    # Fixed widths for numerical stability; centers provide local flexibility
    centers = np.linspace(-2.5, 2.5, 8)
    width = 0.8  # moderate width for good overlap without over-conditioning

    # Build Gaussian basis: shape (n_points, 8)
    x_expanded = x[:, np.newaxis]
    gauss = np.exp(-0.5 * ((x_expanded - centers[np.newaxis, :]) / width) ** 2)

    # params[0]: constant offset
    # params[1]: linear slope
    # params[2:10]: 8 Gaussian weights
    result = p[0] + p[1] * x
    for i in range(8):
        result = result + p[2 + i] * gauss[:, i]

    return result
# EVOLVE-BLOCK-END
