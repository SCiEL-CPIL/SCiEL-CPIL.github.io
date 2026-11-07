"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

# Fixed RBF centers in [0.5, 2]^3 - corners, face centers, and body center
_CENTERS = np.array([
    [0.5,  0.5,  0.5],
    [2.0,  2.0,  2.0],
    [0.5,  2.0,  0.5],
    [2.0,  0.5,  0.5],
    [0.5,  0.5,  2.0],
    [2.0,  2.0,  0.5],
    [0.5,  2.0,  2.0],
    [2.0,  0.5,  2.0],
    [1.25, 1.25, 1.25],
    [1.25, 0.5,  0.5],
], dtype=np.float64)
_SIGMA2 = 2.0 * 0.65 ** 2  # 2 * sigma^2 for Gaussian kernel


def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    10-parameter RBF model with fixed centers and widths.
    params: weights for each RBF basis function.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    p = params

    # Compute squared distances from each point to each center
    dx = x[:, np.newaxis] - _CENTERS[:, 0][np.newaxis, :]
    dy = y[:, np.newaxis] - _CENTERS[:, 1][np.newaxis, :]
    dz = z[:, np.newaxis] - _CENTERS[:, 2][np.newaxis, :]
    dist_sq = dx * dx + dy * dy + dz * dz  # (n, 10)

    # Gaussian RBF values
    rbf = np.exp(-dist_sq / _SIGMA2)  # (n, 10)

    return rbf @ p
# EVOLVE-BLOCK-END
