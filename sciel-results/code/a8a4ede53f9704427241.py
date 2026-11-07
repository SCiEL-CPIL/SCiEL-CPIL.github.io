"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE = time.time() + 145.0  # safety margin before 150s kill


def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Full quadratic polynomial in centered x, y, z with 10 parameters.
    Inputs are centered at 1.25 (midpoint of [0.5, 2]) for better conditioning.
    params[0]  : constant
    params[1]  : x_c
    params[2]  : y_c
    params[3]  : z_c
    params[4]  : x_c^2
    params[5]  : y_c^2
    params[6]  : z_c^2
    params[7]  : x_c*y_c
    params[8]  : x_c*z_c
    params[9]  : y_c*z_c
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    p = params

    # Center inputs for better numerical conditioning
    xc = x - 1.25
    yc = y - 1.25
    zc = z - 1.25

    return (p[0] +
            p[1] * xc +
            p[2] * yc +
            p[3] * zc +
            p[4] * xc * xc +
            p[5] * yc * yc +
            p[6] * zc * zc +
            p[7] * xc * yc +
            p[8] * xc * zc +
            p[9] * yc * zc)
# EVOLVE-BLOCK-END
