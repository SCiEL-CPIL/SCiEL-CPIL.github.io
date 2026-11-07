# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form symbolic regression with 2 inputs (x, y) and 10 free params.
    Uses a rich basis: linear, quadratic, cubic, products, trig, and radial
    terms. The evaluator fits params with BFGS from all-ones.
    """
    p = params
    # Ensure x, y are 1-D arrays of same length
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    # Basis functions (10 of them, each a vector over the points)
    # 1. constant
    b0 = np.ones_like(x)
    # 2. x
    b1 = x
    # 3. y
    b2 = y
    # 4. x^2
    b3 = x * x
    # 5. y^2
    b4 = y * y
    # 6. x*y
    b5 = x * y
    # 7. x^3
    b6 = x * x * x
    # 8. y^3
    b7 = y * y * y
    # 9. x^2*y + x*y^2  (cubic mixed)
    b8 = x * x * y + x * y * y
    # 10. radial: x^2 + y^2 (norm squared)
    b9 = x * x + y * y

    # Weighted combination
    pred = (p[0] * b0 + p[1] * b1 + p[2] * b2 + p[3] * b3 + p[4] * b4 +
            p[5] * b5 + p[6] * b6 + p[7] * b7 + p[8] * b8 + p[9] * b9)

    # Guarantee finite output
    if not np.all(np.isfinite(pred)):
        pred = np.nan_to_num(pred, nan=0.0, posinf=1e6, neginf=-1e6)

    return pred
# EVOLVE-BLOCK-END
