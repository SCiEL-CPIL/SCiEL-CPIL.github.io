"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Rich basis function for symbolic regression with 10 parameters.
    Combines polynomial, rational, and exponential terms to capture
    a wide variety of hidden functions.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    p = params

    # Clip inputs to avoid numerical issues with exp and division
    x_safe = np.clip(x, 0.1, 10.0)
    y_safe = np.clip(y, 0.1, 10.0)
    z_safe = np.clip(z, 0.1, 10.0)

    # Basis terms
    term0 = np.ones_like(x)
    term1 = x_safe
    term2 = y_safe
    term3 = z_safe
    term4 = x_safe * y_safe
    term5 = y_safe * z_safe
    term6 = x_safe * z_safe
    term7 = 1.0 / (x_safe + 0.1)
    term8 = np.exp(-x_safe)
    term9 = np.sin(y_safe)

    return (p[0] * term0 +
            p[1] * term1 +
            p[2] * term2 +
            p[3] * term3 +
            p[4] * term4 +
            p[5] * term5 +
            p[6] * term6 +
            p[7] * term7 +
            p[8] * term8 +
            p[9] * term9)
# EVOLVE-BLOCK-END
