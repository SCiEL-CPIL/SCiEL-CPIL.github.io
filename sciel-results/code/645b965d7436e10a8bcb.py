"""Refined Cohn-Goncalves Laguerre construction for the Fourier sign uncertainty bound.

Optimizes the 12 double-root locations (m=12, n=1) to minimize the C4 upper bound,
which is governed by the last sign change (the largest root).  Roots are kept
positive and ordered via a log-space parameterization and SLSQP bounds; Laguerre
orthogonality/positivity conditions are enforced through a smooth penalty.
"""
import time
import numpy as np
from scipy.optimize import minimize

_M = 12          # number of double roots
_N = 1           # dimension
_DEADLINE = 560.0   # safety margin well below the 600 s wall-clock limit


def _laguerre(n, x):
    """Generalized Laguerre polynomial L_n^(alpha) with alpha = 2n_dim (n_dim=1)."""
    alpha = 2.0 * _N
    # Recurrence: L_0=1, L_1=1+alpha-x
    if n == 0:
        return np.ones_like(x, dtype=float)
    L0 = np.ones_like(x, dtype=float)
    L1 = (1.0 + alpha) - x
    if n == 1:
        return L1
    for k in range(1, n):
        L2 = ((2 * k + 1 + alpha - x) * L1 - (k + alpha) * L0) / (k + 1)
        L0, L1 = L1, L2
    return L1


def _bound_from_roots(roots):
    """C4 upper bound ~ last sign change = max root (monotone in r_max)."""
    return float(np.max(roots))


def _penalty(roots):
    """Smooth penalty for violating Laguerre orthogonality/positivity constraints."""
    r = np.asarray(roots, dtype=float)
    pen = 0.0
    # positivity margin (roots should stay comfortably > 0)
    pen += 1e4 * float(np.sum(np.maximum(0.0, 0.5 - r)))
    # spacing: discourage roots collapsing together (double roots need separation)
    d = np.diff(np.sort(r))
    pen += 1e3 * float(np.sum(np.maximum(0.0, 0.1 - d)))
    # Laguerre orthogonality surrogate: the weighted polynomial should have the
    # correct number of sign changes; penalize residual of orthogonality moments.
    # Use a low-order moment check on the product polynomial P(r)=prod (x - r_i).
    x = np.linspace(0.0, float(np.max(r)) * 1.2, 40)
    w = x ** (2 * _N) * np.exp(-x)
    P = np.ones_like(x)
    for ri in r:
        P = P * (x - ri)
    # orthogonality: <P, L_k> should be small for k < m (approx); penalize magnitude
    for k in range(0, 3):
        Lk = _laguerre(k, x)
        moment = float(np.trapz(P * Lk * w, x))
        pen += 1e-3 * moment * moment
    return pen


def _objective(x):
    """Objective in log-space: bound + penalty.  x_i = log(r_i)."""
    r = np.exp(x)
    return _bound_from_roots(r) + _penalty(r)


def _initial_roots():
    # Seed: arithmetic progression (matches the seed program) in log-space.
    return np.log(np.array([4.0 * (i + 1) for i in range(_M)], dtype=float))


def construct():
    start = time.time()
    x0 = _initial_roots()
    bounds = [(0.0, 10.0)] * _M   # r in (1, e^10); positivity guaranteed

    best_x = x0.copy()
    best_val = _objective(best_x)
    best_roots = np.exp(best_x)

    def _elapsed_ok():
        return (time.time() - start) < _DEADLINE

    # Multi-start SLSQP with a few perturbations, under the deadline guard.
    rng = np.random.default_rng(0)
    for trial in range(6):
        if not _elapsed_ok():
            break
        if trial == 0:
            x_try = x0.copy()
        else:
            x_try = x0 + rng.normal(0.0, 0.25, size=_M)
            x_try = np.clip(x_try, 0.0, 10.0)
        try:
            res = minimize(
                _objective,
                x_try,
                method="SLSQP",
                bounds=bounds,
                options={"maxiter": 200, "ftol": 1e-9},
            )
            if _elapsed_ok() and res.x is not None:
                val = _objective(res.x)
                if val < best_val:
                    best_val = val
                    best_x = res.x.copy()
                    best_roots = np.exp(best_x)
        except Exception:
            continue

    # Final high-precision polish pass on the best solution found.
    if _elapsed_ok():
        try:
            res = minimize(
                _objective,
                best_x,
                method="SLSQP",
                bounds=bounds,
                options={"maxiter": 500, "ftol": 1e-12},
            )
            if res.x is not None:
                val = _objective(res.x)
                if val < best_val:
                    best_val = val
                    best_x = res.x.copy()
                    best_roots = np.exp(best_x)
        except Exception:
            pass

    roots = np.sort(np.exp(best_x))
    return [float(v) for v in roots]


if __name__ == "__main__":
    print(construct())
