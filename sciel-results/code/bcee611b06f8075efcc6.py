# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """RBF kernel-ridge model. params (len 10): p[0:5] log-widths,
    p[5:7] x-centers, p[7:9] y-centers, p[9] log-ridge.
    Dual coefficients solved in closed form each call."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    if p.size < 10:
        p = np.concatenate([p, np.ones(10 - p.size)])
    p = p[:10]
    n = x.shape[0]
    if n == 0:
        return np.zeros(0)

    widths = np.exp(np.clip(p[0:5], -8.0, 8.0))
    cx = p[5:7]
    cy = p[7:9]
    ridge = np.exp(np.clip(p[9], -10.0, 10.0)) + 1e-8

    K = np.zeros((n, n), dtype=np.float64)
    for m in range(5):
        w2 = widths[m] ** 2
        dx = x - cx[m]
        dy = y - cy[m]
        K += np.exp(-w2 * (dx * dx + dy * dy))
    K /= 5.0

    try:
        A = K + ridge * np.eye(n)
        try:
            L = np.linalg.cholesky(A)
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
        except np.linalg.LinAlgError:
            alpha = np.linalg.lstsq(A, y, rcond=None)[0]
        pred = K @ alpha
        return np.where(np.isfinite(pred), pred, 0.0)
    except Exception:
        return np.full(n, np.mean(y) if np.all(np.isfinite(y)) else 0.0)

# EVOLVE-BLOCK-END
