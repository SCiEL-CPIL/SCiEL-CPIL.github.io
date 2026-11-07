"""Symbolic regression with flexible basis; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Flexible model with 10 parameters using a diverse basis:
    - Polynomial terms (even/odd powers)
    - Rational terms (1/(1+x^2), x/(1+x^2))
    - Bounded nonlinear terms (tanh, arctan)
    - Gaussian-like bump

    This basis can approximate many smooth functions on [-2, 2].
    """
    x = np.asarray(x, dtype=float)

    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params

    x2 = x * x
    x3 = x2 * x
    x4 = x2 * x2
    x5 = x4 * x

    # Denominators for rational terms (avoid division by zero)
    denom1 = 1.0 + x2
    denom2 = 1.0 + 0.5 * x2

    # Basis functions
    b0 = 1.0
    b1 = x
    b2 = x2
    b3 = x3
    b4 = x4
    b5 = x5
    b6 = 1.0 / denom1
    b7 = x / denom1
    b8 = np.tanh(x)
    b9 = np.sin(x)

    result = (p0 * b0 + p1 * b1 + p2 * b2 + p3 * b3 + p4 * b4 +
              p5 * b5 + p6 * b6 + p7 * b7 + p8 * b8 + p9 * b9)

    return result
# EVOLVE-BLOCK-END
