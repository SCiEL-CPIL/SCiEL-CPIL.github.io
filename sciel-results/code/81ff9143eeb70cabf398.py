"""
Symbolic regression with 2 input(s).
Uses a rich set of basis functions to approximate the unknown target function.
The 10 parameters are fitted by the evaluator via BFGS.
"""
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Linear combination of 10 basis functions of (x, y).
    Basis:
      0: 1
      1: x
      2: y
      3: x^2
      4: y^2
      5: x*y
      6: tanh(x)
      7: tanh(y)
      8: x^3
      9: y^3
    """
    p = params
    x = np.asarray(x, dtype=np.float64).flatten()
    y = np.asarray(y, dtype=np.float64).flatten()

    # Compute basis functions
    b0 = np.ones_like(x)
    b1 = x
    b2 = y
    b3 = x * x
    b4 = y * y
    b5 = x * y
    b6 = np.tanh(x)
    b7 = np.tanh(y)
    b8 = x * x * x
    b9 = y * y * y

    pred = (
        p[0] * b0 +
        p[1] * b1 +
        p[2] * b2 +
        p[3] * b3 +
        p[4] * b4 +
        p[5] * b5 +
        p[6] * b6 +
        p[7] * b7 +
        p[8] * b8 +
        p[9] * b9
    )
    return pred
