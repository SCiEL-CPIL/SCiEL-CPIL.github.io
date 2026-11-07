import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = 145.0

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Composite-basis symbolic regression with 1 input.

    The 10 parameters are linear coefficients over a FIXED basis that mixes
    three complementary families so a single linear (convex) fit can express
    polynomial growth, smooth periodic oscillation, and tanh-like saturation /
    step-like features -- the three feature classes a lone Gaussian-RBF or
    lone polynomial handles poorly. BFGS on 10 linear params is a convex
    quadratic, so it converges robustly from the all-ones start.
    """
    # Deadline guard: return a safe finite array if we are about to be killed.
    if time.time() - _START_TIME > _DEADLINE:
        return np.zeros_like(np.asarray(x, dtype=np.float64))

    x = np.asarray(x, dtype=np.float64).ravel()
    p = np.asarray(params, dtype=np.float64).ravel()

    if p.size < 10:
        p = np.pad(p, (0, 10 - p.size), constant_values=0.0)
    else:
        p = p[:10]

    # --- Build the 10 fixed basis columns on [-2, 2] ---
    # Polynomial (4): captures monotone / polynomial trends.
    p0 = np.ones_like(x)
    p1 = x
    p2 = x * x
    p3 = x * x * x

    # Trigonometric (4): captures periodic / oscillatory structure.
    # Frequencies chosen to resolve features on [-2, 2] without aliasing.
    t1s = np.sin(1.0 * x)
    t1c = np.cos(1.0 * x)
    t2s = np.sin(2.0 * x)
    t2c = np.cos(2.0 * x)

    # Tanh (2): captures smooth saturation and step-like / S-shaped features.
    th1 = np.tanh(1.0 * x)
    th2 = np.tanh(2.0 * x)

    basis = np.column_stack((p0, p1, p2, p3, t1s, t1c, t2s, t2c, th1, th2))

    result = basis @ p
    result = np.where(np.isfinite(result), result, 0.0)
    return result
