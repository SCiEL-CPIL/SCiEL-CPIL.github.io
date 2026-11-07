# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression candidate with a fixed, smooth 10-term basis.
    BFGS (started at all ones) fits the 10 coefficients to the training data.

    Basis (closed-form, smooth, finite on [-1,1]^2):
      1. 1
      2. x
      3. y
      4. x^2
      5. y^2
      6. x*y
      7. x^3
      8. y^3
      9. sin(x) * cos(y)
     10. cos(x) * sin(y)
    This spans polynomials up to degree 3 (including the quadratic interaction)
    plus smooth oscillatory components, giving a broad class of recoverable
    target functions while keeping the fit fast and numerically stable.
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    out = (
        p0
        + p1 * x
        + p2 * y
        + p3 * x * x
        + p4 * y * y
        + p5 * x * y
        + p6 * x * x * x
        + p7 * y * y * y
        + p8 * np.sin(x) * np.cos(y)
        + p9 * np.cos(x) * np.sin(y)
    )
    return out
# EVOLVE-BLOCK-END
