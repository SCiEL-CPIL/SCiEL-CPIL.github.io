"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Duffing-van der Pol oscillator model.
    a = p1*x + p2*x^3 - p3*v - p4*v^3 + p5*x*v^2 + p6*v^2*x^2 + p7*x^2*v + p8*x^4 + p9*v^4 + p10
    """
    x2 = x * x
    x3 = x2 * x
    x4 = x2 * x2
    v2 = v * v
    v3 = v2 * v
    v4 = v2 * v2

    return (params[0] * x +
            params[1] * x3 -
            params[2] * v -
            params[3] * v3 +
            params[4] * x * v2 +
            params[5] * v2 * x2 +
            params[6] * x2 * v +
            params[7] * x4 +
            params[8] * v4 +
            params[9])
# EVOLVE-BLOCK-END
