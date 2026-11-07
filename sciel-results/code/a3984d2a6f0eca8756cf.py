"""Symbolic regression candidate: cubic (10-term) basis in x, y, z.

The evaluator fits 10 scalar params via BFGS from ones, minimizing training MSE.
We supply a cubic polynomial basis that includes all terms up to degree 3
that are most likely to capture smooth nonlinear structure. The basis is:
1, x, y, z, x^2, y^2, z^2, x^3, y^3, z^3

This allows the model to capture pure cubic curvature in each variable while
maintaining numerical stability for inputs in [0.5, 2]. The lower-order terms
allow degeneration to simpler models if needed.
"""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)

    # 10-term basis: 1, x, y, z, x^2, y^2, z^2, x^3, y^3, z^3
    c1 = np.ones_like(x, dtype=np.float64)
    c2 = x
    c3 = y
    c4 = z
    c5 = x * x
    c6 = y * y
    c7 = z * z
    c8 = x * x * x
    c9 = y * y * y
    c10 = z * z * z

    out = (
        p[0] * c1
        + p[1] * c2
        + p[2] * c3
        + p[3] * c4
        + p[4] * c5
        + p[5] * c6
        + p[6] * c7
        + p[7] * c8
        + p[8] * c9
        + p[9] * c10
    )
    return out
# EVOLVE-BLOCK-END
