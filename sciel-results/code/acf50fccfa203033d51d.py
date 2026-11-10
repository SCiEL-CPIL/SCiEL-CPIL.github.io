# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

_N = 64
_A = (np.arange(_N, dtype=np.float64) + 1) / _N  # a_i = (i+1)/64, shape (64,)
_T0 = time.time()
_BUDGET = 500.0  # seconds; keep a margin under the 600 s limit


def _fast_area(x, M=2000):
    """Area surrogate using M quadrature heights (fast, for optimization)."""
    y = (np.arange(M) + 0.5) / M
    w = (1.0 - y) / _N                       # (M,)
    L = x[None, :] + _A[None, :] * y[:, None]  # (M, 64)
    Ls = np.sort(L, axis=1)                    # (M, 64)
    gaps = np.diff(Ls, axis=1)                 # (M, 63)
    ul = w[:, None] + np.minimum(gaps, w[:, None]).sum(axis=1)  # (M,)
    return float(ul.mean())


def _true_area(x):
    """Exact area using the contract's 20000 quadrature heights."""
    y = (np.arange(20000) + 0.5) / 20000.0
    w = (1.0 - y) / _N
    L = x[None, :] + _A[None, :] * y[:, None]
    Ls = np.sort(L, axis=1)
    gaps = np.diff(Ls, axis=1)
    ul = w[:, None] + np.minimum(gaps, w[:, None]).sum(axis=1)
    return float(ul.mean())


def _coord_descent(x, rounds=4, steps=(0.2, 0.05, 0.01)):
    """Non-smooth-friendly coordinate descent on the fast area."""
    x = x.copy()
    cur = _fast_area(x)
    for _ in range(rounds):
        improved = False
        for i in range(_N):
            for s in steps:
                for d in (1.0, -1.0):
                    trial = x.copy()
                    trial[i] += d * s
                    v = _fast_area(trial)
                    if v < cur - 1e-15:
                        x = trial
                        cur = v
                        improved = True
        if not improved:
            break
    return x, cur


def kakeya_needle_2d_positions64() -> np.ndarray:
    """
    Minimize the union area of 64 Kakeya-needle triangles by direct numerical
    optimization (a different approach family from the closed-form Keich
    construction): Powell's method on a coarse quadrature surrogate, followed by
    a coordinate-descent polish, with several diverse restarts. The best
    candidate is selected by the exact 20000-height area.
    """
    rng = np.random.default_rng(0)
    inits = [
        np.zeros(_N),
        -_A.copy(),
        -0.5 * _A.copy(),
        rng.normal(0.0, 0.1, _N),
    ]

    best_x = None
    best_true = np.inf

    for init in inits:
        if time.time() - _T0 > _BUDGET:
            break
        # Global (derivative-free) search on the fast surrogate.
        try:
            res = minimize(_fast_area, init, method="Powell",
                           options={"maxiter": 120, "maxfev": 3000,
                                    "xtol": 1e-6, "ftol": 1e-8})
            cand = np.asarray(res.x, dtype=np.float64)
        except Exception:
            cand = np.asarray(init, dtype=np.float64).copy()

        # Local coordinate-descent polish (robust to the non-smooth sorting).
        cand, _ = _coord_descent(cand)

        if not np.all(np.isfinite(cand)):
            continue

        ta = _true_area(cand)
        if ta < best_true:
            best_true = ta
            best_x = cand.copy()

    if best_x is None:
        best_x = np.zeros(_N)
    return best_x.astype(np.float64)


# EVOLVE-BLOCK-END
