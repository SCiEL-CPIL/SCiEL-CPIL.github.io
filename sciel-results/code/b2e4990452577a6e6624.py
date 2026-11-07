"""Rewritten program with a richer, well-conditioned basis for symbolic regression."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Approximate an unknown function y = f(x) for x in [-2, 2] using a linear
    combination of 10 well-conditioned basis functions. The parameters are fit
    by BFGS starting from ones.

    Basis:
      1. 1
      2. x
      3. x^2
      4. x^3
      5. x^4
      6. sin(pi * x / 2)   -> zero at x = -2, 0, 2
      7. cos(pi * x / 2)   -> 1 at x = 0, 0 at x = ±2
      8. tanh(x)
      9. x * tanh(x)
      10. x^2 * tanh(x)
    """
    p = params

    # Guard against non-finite params during optimization
    if not np.all(np.isfinite(p)):
        p = np.ones(10)

    # Ensure x is a 1-D array
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 0:
        x = x.reshape(1)

    # Basis 1-5: polynomials up to degree 4
    b1 = np.ones_like(x)
    b2 = x
    b3 = x ** 2
    b4 = x ** 3
    b5 = x ** 4

    # Basis 6-7: trigonometric
    angle = np.pi * x / 2.0
    b6 = np.sin(angle)
    b7 = np.cos(angle)

    # Basis 8-10: tanh-based
    th = np.tanh(x)
    b8 = th
    b9 = x * th
    b10 = x ** 2 * th

    # Combine
    pred = (p[0] * b1 + p[1] * b2 + p[2] * b3 + p[3] * b4 + p[4] * b5 +
            p[5] * b6 + p[6] * b7 + p[7] * b8 + p[8] * b9 + p[9] * b10)

    return pred
# EVOLVE-BLOCK-END
