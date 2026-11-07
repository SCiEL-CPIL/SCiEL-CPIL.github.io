# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Deadline guard
    start_time = time.time()
    deadline = 145.0  # 5s margin before 150s kill

    x = np.asarray(x, dtype=np.float64).flatten()
    n = len(x)
    p = np.asarray(params, dtype=np.float64)
    if p.shape[0] != 10:
        p = p[:10] if p.size >= 10 else np.pad(p, (0, 10 - p.size))

    # Check deadline
    if time.time() > deadline:
        return np.zeros(n, dtype=np.float64)

    # STRUCTURAL CHANGE: Use a hybrid basis that combines:
    # 1) Smooth local basis (RBF-like with fixed centers) for local flexibility
    # 2) Global polynomial trend (low degree) for overall shape
    # 3) A couple of trigonometric terms for potential periodic components
    #
    # This is still a LINEAR model in params (BFGS converges to global optimum),
    # but the basis is more expressive than pure polynomial or pure trig.
    #
    # Basis design (10 functions):
    # - 5 RBF-like bumps at fixed centers (local flexibility)
    # - 3 polynomial terms: 1, x, x^2 (global trend, normalized)
    # - 2 trig terms: sin(x), cos(x) (potential oscillation)
    #
    # Total: 5 + 3 + 2 = 10 basis functions

    xn = x / 3.0  # Normalize to [-1, 1]

    b = np.ones((10, n), dtype=np.float64)

    # 5 RBF-like basis functions (Gaussian bumps) at fixed centers
    # Centers spread across [-1, 1]
    centers = np.linspace(-0.8, 0.8, 5)
    sigma = 0.45  # Width of each Gaussian

    for i in range(5):
        diff = xn - centers[i]
        b[i, :] = np.exp(-(diff ** 2) / (2.0 * sigma ** 2))

    # 3 polynomial basis terms (normalized)
    b[5, :] = 1.0
    b[6, :] = xn
    b[7, :] = xn ** 2

    # 2 trigonometric basis terms
    b[8, :] = np.sin(np.pi * xn)
    b[9, :] = np.cos(np.pi * xn)

    # Linear combination
    result = b @ p

    # Check for non-finite values
    if not np.all(np.isfinite(result)):
        result = np.nan_to_num(result, nan=0.0, posinf=1e10, neginf=-1e10)

    return result
# EVOLVE-BLOCK-END
