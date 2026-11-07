"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    10-parameter reciprocal/interaction basis for rational-structure target.
    params[0]  : 1/x
    params[1]  : 1/y
    params[2]  : 1/z
    params[3]  : 1/(x*y)
    params[4]  : 1/(x*z)
    params[5]  : 1/(y*z)
    params[6]  : 1/(x*y*z)
    params[7]  : x/(y*z)
    params[8]  : y/(x*z)
    params[9]  : z/(x*y)
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    p = params

    inv_x = 1.0 / x
    inv_y = 1.0 / y
    inv_z = 1.0 / z

    return (p[0] * inv_x +
            p[1] * inv_y +
            p[2] * inv_z +
            p[3] * inv_x * inv_y +
            p[4] * inv_x * inv_z +
            p[5] * inv_y * inv_z +
            p[6] * inv_x * inv_y * inv_z +
            p[7] * x * inv_y * inv_z +
            p[8] * y * inv_x * inv_z +
            p[9] * z * inv_x * inv_y)
# EVOLVE-BLOCK-END
