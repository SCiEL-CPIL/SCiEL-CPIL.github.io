"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a nonlinear oscillator modeled with a Duffing-like basis.
    Uses scaled inputs and polynomial features up to cubic order with cross-terms.
    10 parameters: constant, x, x^2, x^3, v, v^2, v^3, xv, x^2v, xv^2
    """
    # Scale inputs to improve numerical conditioning
    # Training ranges: x in [-0.5728, 0.5275], v in [-0.1754, 0.2821]
    # Use characteristic scales ~0.5 and ~0.3
    X = x / 0.5
    V = v / 0.3

    # Precompute powers and cross-terms
    X2 = X * X
    X3 = X2 * X
    V2 = V * V
    V3 = V2 * V
    XV = X * V
    X2V = X2 * V
    XV2 = X * V2

    # 10 parameters as coefficients
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params

    # Construct the polynomial expression
    result = (p0 + p1 * X + p2 * X2 + p3 * X3 +
              p4 * V + p5 * V2 + p6 * V3 +
              p7 * XV + p8 * X2V + p9 * XV2)

    return result
# EVOLVE-BLOCK-END
