"""Symbolic regression using a rich deterministic basis of polynomial, Fourier, and tanh terms."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Model using a deterministic rich basis.
    Linear in parameters, so BFGS from any start point converges reliably.

    params (length 10):
        p0: constant
        p1: coefficient for x
        p2: coefficient for x^2
        p3: coefficient for x^3
        p4: coefficient for sin(x)
        p5: coefficient for cos(x)
        p6: coefficient for sin(2x)
        p7: coefficient for cos(2x)
        p8: coefficient for tanh(x)
        p9: coefficient for tanh(2x)
    """
    x = np.asarray(x, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)

    # Polynomial terms
    x2 = x * x
    x3 = x2 * x

    # Fourier terms
    sin_x = np.sin(x)
    cos_x = np.cos(x)
    sin_2x = np.sin(2 * x)
    cos_2x = np.cos(2 * x)

    # Saturating non-linearities
    tanh_x = np.tanh(x)
    tanh_2x = np.tanh(2 * x)

    # Combine all terms linearly
    result = (
        params[0]
        + params[1] * x
        + params[2] * x2
        + params[3] * x3
        + params[4] * sin_x
        + params[5] * cos_x
        + params[6] * sin_2x
        + params[7] * cos_2x
        + params[8] * tanh_x
        + params[9] * tanh_2x
    )

    # Ensure finite output
    return np.nan_to_num(result, nan=0.0, posinf=1e8, neginf=-1e8)
# EVOLVE-BLOCK-END
