"""Rewritten symbolic regression candidate: rich nonlinear basis with 10 free params."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form expression using a fixed set of 10 basis functions.
    params: length-10 array of coefficients (fitted by BFGS).
    Returns an array with one prediction per point.
    """
    # Unpack parameters
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params

    # Ensure inputs are arrays
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    # Basis functions: diverse nonlinear terms to span common target functions
    # 1. constant
    # 2. x
    # 3. y
    # 4. x^2
    # 5. y^2
    # 6. x*y
    # 7. x^3
    # 8. sin(x)
    # 9. cos(y)
    # 10. tanh(x + y)

    b0 = np.ones_like(x)
    b1 = x
    b2 = y
    b3 = x * x
    b4 = y * y
    b5 = x * y
    b6 = x * x * x
    b7 = np.sin(x)
    b8 = np.cos(y)
    b9 = np.tanh(x + y)

    pred = (
        p0 * b0
        + p1 * b1
        + p2 * b2
        + p3 * b3
        + p4 * b4
        + p5 * b5
        + p6 * b6
        + p7 * b7
        + p8 * b8
        + p9 * b9
    )

    return pred
# EVOLVE-BLOCK-END
