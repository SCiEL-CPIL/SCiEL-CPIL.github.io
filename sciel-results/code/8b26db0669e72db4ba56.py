# EVOLVE-BLOCK-START
import numpy as np
import time

_START = time.time()
_DEADLINE = 150.0
_MARGIN = 5.0


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Deadline guard: this model is O(n) with no loops, so it is fast,
    # but we still respect the wall-clock budget explicitly.
    if time.time() > _DEADLINE - _MARGIN:
        # Return a safe finite fallback (mean of y) to avoid invalid output.
        return np.full_like(x, float(np.mean(y)), dtype=np.float64)

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)

    # Structural change: hybrid basis = full degree-2 polynomial (6 terms)
    # + low-order trigonometric terms (4 terms) = 10 parameters.
    # This spans a wider class of closed-form symbolic functions
    # (polynomial AND periodic) than a pure degree-3 monomial basis,
    # and remains linear in params so BFGS reduces to a well-posed fit.

    b0 = np.ones_like(x)
    b1 = x
    b2 = y
    b3 = x * x
    b4 = y * y
    b5 = x * y

    b6 = np.sin(x)
    b7 = np.sin(y)
    b8 = np.cos(x)
    b9 = np.cos(y)

    pred = (p[0] * b0 +
            p[1] * b1 +
            p[2] * b2 +
            p[3] * b3 +
            p[4] * b4 +
            p[5] * b5 +
            p[6] * b6 +
            p[7] * b7 +
            p[8] * b8 +
            p[9] * b9)

    # Guard against non-finite outputs.
    if not np.all(np.isfinite(pred)):
        pred = np.where(np.isfinite(pred), pred, float(np.mean(y)))

    return pred
# EVOLVE-BLOCK-END
