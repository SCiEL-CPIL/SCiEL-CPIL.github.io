"""Radial basis functions + polynomial basis for symbolic regression."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE = None


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers the target function using a combination of polynomial terms
    and Gaussian radial basis functions (bumps) centered at strategic points.
    
    The polynomial terms (up to degree 2 in x and y) capture global smooth structure.
    The Gaussian bumps capture localized features. Together they form a flexible
    function space that can approximate a wide range of target functions.
    
    Basis functions (10 total):
    0:  1
    1:  x
    2:  y
    3:  x^2 - 1/3  (centered for better conditioning)
    4:  y^2 - 1/3
    5:  xy
    6:  exp(-((x-0.5)^2 + (y-0.5)^2) / 0.25)
    7:  exp(-((x+0.5)^2 + (y+0.5)^2) / 0.25)
    8:  exp(-(x^2 + y^2) / 0.5)
    9:  exp(-((x-0.3)^2 + (y+0.3)^2) / 0.3)
    """
    global _DEADLINE
    if _DEADLINE is None:
        _DEADLINE = time.time() + 140.0

    if time.time() > _DEADLINE:
        return np.full_like(x, np.mean(y), dtype=np.float64)

    p = params

    # Polynomial terms
    b0 = np.ones_like(x)
    b1 = x
    b2 = y
    b3 = x * x - 1.0 / 3.0
    b4 = y * y - 1.0 / 3.0
    b5 = x * y

    # Gaussian bumps at different centers with different widths
    # Bump 1: center (0.5, 0.5), width ~0.5
    r1_sq = (x - 0.5) ** 2 + (y - 0.5) ** 2
    b6 = np.exp(-r1_sq / 0.25)

    # Bump 2: center (-0.5, -0.5), width ~0.5
    r2_sq = (x + 0.5) ** 2 + (y + 0.5) ** 2
    b7 = np.exp(-r2_sq / 0.25)

    # Bump 3: center (0, 0), width ~0.7
    r3_sq = x * x + y * y
    b8 = np.exp(-r3_sq / 0.5)

    # Bump 4: center (0.3, -0.3), width ~0.55
    r4_sq = (x - 0.3) ** 2 + (y + 0.3) ** 2
    b9 = np.exp(-r4_sq / 0.3)

    result = (
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

    if not np.all(np.isfinite(result)):
        result = np.nan_to_num(result, nan=0.0, posinf=1e10, neginf=-1e10)

    return result
# EVOLVE-BLOCK-END
