"""RBF with convex linear solve for weights + small nonlinear width refinement."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

_T0 = time.time()
_DEADLINE = 120.0


def _rbf_basis(x, centers, sigma):
    x = np.asarray(x, dtype=float).reshape(-1)
    d2 = (x[:, None] - centers[None, :]) ** 2
    B = np.exp(-d2 / (2.0 * sigma * sigma))
    B = np.concatenate([B, np.ones((x.size, 1)) / (1.0 + sigma * sigma)], axis=1)
    return B


def _fit(x, y, centers, sigma):
    B = _rbf_basis(x, centers, sigma)
    w, *_ = np.linalg.lstsq(B, y, rcond=None)
    return w


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(params, dtype=float).reshape(-1)
    N = x.size
    if N < 2:
        return np.full(N, float(np.mean(y))) if N else np.array([])

    centers = np.linspace(-2.0, 2.0, 10)

    # Phase 1: convex linear least-squares at a default width (globally optimal
    # for these fixed bases) — robust baseline, no local minima.
    sigma0 = 0.7
    w = _fit(x, y, centers, sigma0)
    pred = _rbf_basis(x, centers, sigma0) @ w
    best_loss = float(np.mean((pred - y) ** 2))
    best_sigma, best_w = sigma0, w

    # Phase 2: refine width sigma with BFGS. For fixed sigma the sub-problem is
    # linear in weights (convex), so this only has to optimize one scalar.
    def obj(s):
        s = float(s)
        if not np.isfinite(s) or s <= 1e-3:
            return 1e30
        wv = _fit(x, y, centers, s)
        p = _rbf_basis(x, centers, s) @ wv
        return float(np.mean((p - y) ** 2))

    if time.time() < _DEADLINE - 30.0:
        try:
            res = minimize(obj, [sigma0], method="BFGS",
                           options={"maxiter": 60, "gtol": 1e-6})
            if res.success or np.isfinite(res.fun):
                s = float(np.clip(res.x[0], 0.05, 5.0))
                wv = _fit(x, y, centers, s)
                p = _rbf_basis(x, centers, s) @ wv
                loss = float(np.mean((p - y) ** 2))
                if np.isfinite(loss) and loss < best_loss:
                    best_loss, best_sigma, best_w = loss, s, wv
        except Exception:
            pass

    pred = _rbf_basis(x, centers, best_sigma) @ best_w
    if not np.all(np.isfinite(pred)):
        pred = np.where(np.isfinite(pred), pred, float(np.mean(y)))
    return pred
# EVOLVE-BLOCK-END
