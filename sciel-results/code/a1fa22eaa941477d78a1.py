# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

N = 64
W = 1.0 / N
A = np.arange(1, N + 1) / N  # a_i for i=0..63
M = 20000
Y = (np.arange(M) + 0.5) / M  # heights (M,)
DEADLINE = 600.0  # seconds, safety margin before 630s kill


def _union_area(x):
    """Compute union area of the 64 triangles via quadrature (vectorized)."""
    # L_i(y) = x_i + a_i * y, shape (N, M)
    L = x[:, None] + A[:, None] * Y[None, :]
    L.sort(axis=0)  # sort L_i at each height
    # union length = w + sum over consecutive pairs of min(L_{j+1}-L_j, w)
    gaps = np.diff(L, axis=0)  # (N-1, M)
    union_len = W + np.minimum(gaps, W).sum(axis=0)  # (M,)
    return float(union_len.mean())


def _keich_positions():
    """Classical Keich construction for n=64 (power of 2)."""
    k = int(np.log2(N))
    j = np.arange(1, k + 1)
    positions = np.zeros(N)
    for i in range(N):
        eps = np.array([int(c) for c in bin(i)[2:]], dtype=np.float64)
        eps = np.pad(eps, (k - len(eps), 0), "constant")
        positions[i] = np.sum((1.0 - j) * eps * (2.0 ** -j) / k)
    return positions


def _refine(x0, t_start, time_budget):
    """Refine x0 with L-BFGS-B, respecting deadline. Returns best x found."""
    best_x = x0.copy()
    best_area = _union_area(x0)
    t = time.time()

    # L-BFGS-B on the smooth-ish objective
    res = minimize(
        _union_area,
        x0,
        method="L-BFGS-B",
        options={"maxiter": 2000, "maxfun": 20000, "ftol": 1e-14, "gtol": 1e-10,
                 "disp": False},
    )
    if res.x is not None and np.all(np.isfinite(res.x)):
        a = _union_area(res.x)
        if a < best_area:
            best_area = a
            best_x = res.x.copy()
    t = time.time()
    if t - t_start > time_budget:
        return best_x
    return best_x


def kakeya_needle_2d_positions64() -> np.ndarray:
    """
    Return 64 finite floats x_0..x_63 minimizing the union area of the
    Kakeya needle triangles (n=64). Uses Keich construction as a strong
    initialization, then refines with gradient-based optimization across
    multiple restarts, all bounded by a wall-clock deadline.
    """
    t_start = time.time()
    time_budget = DEADLINE - 10.0  # 10s safety margin

    # Strong seed: Keich construction
    keich = _keich_positions()
    best_x = keich.copy()
    best_area = _union_area(keich)

    # Refine the Keich construction
    t = time.time()
    x_ref = _refine(keich, t_start, time_budget)
    if x_ref is not None and np.all(np.isfinite(x_ref)):
        a = _union_area(x_ref)
        if a < best_area:
            best_area = a
            best_x = x_ref.copy()

    # Multi-restart: try a few perturbed starts to escape local minima
    rng = np.random.default_rng(12345)
    t = time.time()
    remaining = time_budget - (t - t_start)
    if remaining > 15.0:
        n_restarts = max(1, int(remaining / 20.0))
        for r in range(n_restarts):
            if time.time() - t_start > time_budget:
                break
            # Perturb the best solution found so far
            scale = 0.5 / (r + 1)
            x0 = best_x + rng.normal(0.0, scale, size=N)
            t = time.time()
            x_ref = _refine(x0, t_start, time_budget)
            if x_ref is not None and np.all(np.isfinite(x_ref)):
                a = _union_area(x_ref)
                if a < best_area:
                    best_area = a
                    best_x = x_ref.copy()

    # Final validation: ensure finite, correct size, positive area
    best_x = np.asarray(best_x, dtype=np.float64).ravel()
    if best_x.shape != (N,):
        best_x = _keich_positions()
    if not np.all(np.isfinite(best_x)):
        best_x = _keich_positions()
    area = _union_area(best_x)
    if not np.isfinite(area) or area <= 0:
        best_x = _keich_positions()

    return best_x


# EVOLVE-BLOCK-END
