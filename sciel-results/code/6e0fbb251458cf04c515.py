import numpy as np
import time

# Chebyshev polynomial basis mapped to [0.5, 2]
# We have 10 free parameters -> use 10 basis functions
# The evaluator fits params via BFGS starting from all ones.

def _chebyshev(t, degree):
    """Compute Chebyshev polynomial T_n(t) for array t using recurrence."""
    n = t.shape[0]
    if degree == 0:
        return np.ones(n, dtype=np.float64)
    if degree == 1:
        return t.copy()
    T0 = np.ones(n, dtype=np.float64)
    T1 = t.copy()
    for k in range(2, degree + 1):
        T2 = 2.0 * t * T1 - T0
        T0 = T1
        T1 = T2
    return T1

def _basis(x, y, z):
    """Return a (N, 10) matrix of Chebyshev basis functions evaluated at each point."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    n = x.shape[0]

    # Map inputs from [0.5, 2] to [-1, 1]
    # t = 2*(v - 0.5) / (2 - 0.5) - 1 = 2*(v - 0.5)/1.5 - 1 = (4/3)*(v - 0.5) - 1
    # Alternatively: t = (2*v - 2.5) / 1.5 = (4*v - 5) / 3
    x_t = (4.0 * x - 5.0) / 3.0
    y_t = (4.0 * y - 5.0) / 3.0
    z_t = (4.0 * z - 5.0) / 3.0

    # Clip to [-1, 1] for numerical stability (inputs are already in [0.5, 2] so this shouldn't be needed)
    x_t = np.clip(x_t, -1.0, 1.0)
    y_t = np.clip(y_t, -1.0, 1.0)
    z_t = np.clip(z_t, -1.0, 1.0)

    # Build 10 basis terms using Chebyshev polynomials
    # Term 0: constant
    t0 = np.ones(n, dtype=np.float64)

    # Term 1: T_1(x) = x_t
    t1 = x_t

    # Term 2: T_1(y) = y_t
    t2 = y_t

    # Term 3: T_1(z) = z_t
    t3 = z_t

    # Term 4: T_2(x) = 2*x_t^2 - 1
    t4 = 2.0 * x_t * x_t - 1.0

    # Term 5: T_2(y) = 2*y_t^2 - 1
    t5 = 2.0 * y_t * y_t - 1.0

    # Term 6: T_2(z) = 2*z_t^2 - 1
    t6 = 2.0 * z_t * z_t - 1.0

    # Term 7: T_3(x) = 4*x_t^3 - 3*x_t
    t7 = 4.0 * x_t**3 - 3.0 * x_t

    # Term 8: T_3(y) = 4*y_t^3 - 3*y_t
    t8 = 4.0 * y_t**3 - 3.0 * y_t

    # Term 9: T_3(z) = 4*z_t^3 - 3*z_t
    t9 = 4.0 * z_t**3 - 3.0 * z_t

    # Stack into (N, 10)
    B = np.column_stack([t0, t1, t2, t3, t4, t5, t6, t7, t8, t9])

    # Ensure finite
    B = np.nan_to_num(B, nan=0.0, posinf=0.0, neginf=0.0)
    return B

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)

    # Guard: ensure params has length 10
    if p.shape[0] != 10:
        p = np.resize(p, 10)

    B = _basis(x, y, z)

    # Linear combination: B @ p
    result = B @ p

    # Ensure finite
    if not np.all(np.isfinite(result)):
        result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)

    return result
