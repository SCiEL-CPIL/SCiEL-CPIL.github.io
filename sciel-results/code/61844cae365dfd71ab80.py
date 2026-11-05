import numpy as np
import time
from scipy.optimize import minimize

N = 400
_EPS = 1e-6  # scale-anchoring penalty strength (negligible effect on value)


def _value_and_grad(h):
    """Return (value + penalty, gradient). value is scale-invariant."""
    S = float(np.sum(h))
    if abs(S) < 1e-12:
        return np.inf, np.zeros_like(h)
    c = np.convolve(h, h)
    m = float(np.max(c))
    jstar = int(np.argmax(c))
    val = 2.0 * N * m / (S * S)

    # dm/dh[i] = 2*h[jstar-i] when 0 <= jstar-i < N else 0
    idx = jstar - np.arange(N)
    valid = (idx >= 0) & (idx < N)
    g = np.zeros(N)
    g[valid] = 2.0 * h[idx[valid]]

    grad_val = 2.0 * N * (g / (S * S) - 2.0 * m / (S ** 3))

    # Tiny penalty to fix the (flat) scale direction -> well-conditioned BFGS
    penalty = 0.5 * _EPS * (S - 1.0) ** 2
    grad_pen = _EPS * (S - 1.0) * np.ones(N)
    return val + penalty, grad_val + grad_pen


def construct():
    t0 = time.time()
    deadline = 100.0  # safety margin well under the 150 s kill

    best_h = np.ones(N, dtype=float)
    best_val, _ = _value_and_grad(best_h)  # seed value = 2.0

    rng = np.random.default_rng(20240601)

    def objective(h):
        v, _ = _value_and_grad(h)
        return v

    def grad(h):
        _, gg = _value_and_grad(h)
        return gg

    n_restarts = 0
    max_restarts = 600
    while n_restarts < max_restarts and (time.time() - t0) < (deadline - 3.0):
        r = n_restarts
        if r == 0:
            h0 = np.ones(N)
        elif r % 5 == 1:
            h0 = rng.standard_normal(N)
        elif r % 5 == 2:
            # smooth random walk (low DC, structured)
            h0 = np.cumsum(rng.standard_normal(N))
            h0 = h0 - np.mean(h0)
        elif r % 5 == 3:
            k = int(rng.integers(1, N // 2))
            h0 = np.cos(2.0 * np.pi * k * np.arange(N) / N)
        else:
            # local perturbation of the current best
            h0 = best_h + 0.15 * rng.standard_normal(N)

        if abs(float(np.sum(h0))) < 1e-6:
            h0 = h0 + 1.0

        try:
            res = minimize(objective, h0, jac=grad, method="L-BFGS-B",
                           options={"maxiter": 150, "maxfun": 400,
                                    "ftol": 1e-14, "gtol": 1e-10})
            if np.isfinite(res.fun) and res.fun < best_val:
                best_val = res.fun
                best_h = res.x.copy()
        except Exception:
            pass
        n_restarts += 1

    # Rescale to sum=1 (value is scale-invariant, so this is safe) for validity
    S = float(np.sum(best_h))
    if abs(S) > 1e-9:
        best_h = best_h / S
    else:
        best_h = best_h + 1.0
    return best_h
