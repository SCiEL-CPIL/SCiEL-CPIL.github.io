"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE = time.time() + 145.0  # safety margin before 150s kill


def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Full quadratic polynomial in x, y, z with 10 parameters.
    Uses a single matvec for numerical stability and speed.
    params[0]  : constant
    params[1]  : x
    params[2]  : y
    params[3]  : z
    params[4]  : x^2
    params[5]  : y^2
    params[6]  : z^2
    params[7]  : x*y
    params[8]  : x*z
    params[9]  : y*z
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)

    n = x.size
    # Build feature matrix: [1, x, y, z, x^2, y^2, z^2, x*y, x*z, y*z]
    # Use column-stacking to avoid repeated temp allocations
    features = np.empty((n, 10), dtype=np.float64)
    features[:, 0] = 1.0
    features[:, 1] = x
    features[:, 2] = y
    features[:, 3] = z
    features[:, 4] = x * x
    features[:, 5] = y * y
    features[:, 6] = z * z
    features[:, 7] = x * y
    features[:, 8] = x * z
    features[:, 9] = y * z

    # Single BLAS dgemv call
    return features @ p
# EVOLVE-BLOCK-END
