# EVOLVE-BLOCK-START
import time
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid basis: 6 Chebyshev polynomial terms + 4 Gaussian RBF terms.
    All 10 parameters are linear coefficients, so BFGS sees a convex quadratic.
    
    The Chebyshev terms capture global polynomial structure up to degree 3
    with good conditioning on [-1,1]. The RBF terms provide local flexibility
    for non-polynomial features. This combines the structural idea of the
    polynomial/RBF inspirations with a well-conditioned representation.
    """
    t0 = time.time()

    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    n = x.size
    p = np.asarray(params, dtype=np.float64).ravel()

    if p.size < 10:
        p = np.pad(p, (0, 10 - p.size))

    # --- Chebyshev polynomial basis (6 terms) ---
    # T0(t) = 1, T1(t) = t, T2(t) = 2t^2 - 1, T3(t) = 4t^3 - 3t
    T0_x = np.ones(n)
    T1_x = x
    T2_x = 2.0 * x**2 - 1.0
    T3_x = 4.0 * x**3 - 3.0 * x

    T0_y = np.ones(n)
    T1_y = y
    T2_y = 2.0 * y**2 - 1.0
    T3_y = 4.0 * y**3 - 3.0 * y

    # 6 polynomial basis functions:
    b0 = T0_x * T0_y          # 1 (bias)
    b1 = T1_x * T0_y          # x
    b2 = T0_x * T1_y          # y
    b3 = T2_x * T1_y          # (2x^2 - 1) * y  -> captures x^2*y
    b4 = T1_x * T2_y          # x * (2y^2 - 1)  -> captures x*y^2
    b5 = T1_x * T1_y          # x * y

    # --- Gaussian RBF basis (4 terms) ---
    # Centers at quadrant centers of [-1,1]^2, sigma^2 = 2.0
    sigma_sq = 2.0

    r1_sq = (x - 0.5)**2 + (y - 0.5)**2
    b6 = np.exp(-sigma_sq * r1_sq)

    r2_sq = (x + 0.5)**2 + (y + 0.5)**2
    b7 = np.exp(-sigma_sq * r2_sq)

    r3_sq = (x - 0.5)**2 + (y - 0.5)**2
    b8 = np.exp(-sigma_sq * r3_sq)

    r4_sq = (x + 0.5)**2 + (y + 0.5)**2
    b9 = np.exp(-sigma_sq * r4_sq)

    # Combine linearly
    out = (p[0] * b0 +
           p[1] * b1 +
           p[2] * b2 +
           p[3] * b3 +
           p[4] * b4 +
           p[5] * b5 +
           p[6] * b6 +
           p[7] * b7 +
           p[8] * b8 +
           p[9] * b9)

    # Safety: replace any non-finite with 0
    if not np.all(np.isfinite(out)):
        out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

    # Deadline guard
    if time.time() - t0 > 140.0:
        return out

    return out
# EVOLVE-BLOCK-END
