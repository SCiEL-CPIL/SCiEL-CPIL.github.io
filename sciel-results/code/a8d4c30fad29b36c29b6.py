"""Second autocorrelation inequality: optimize 50 nonnegative step heights."""
import time
import numpy as np
from scipy.optimize import minimize

N = 50
M = 2 * N - 1  # 99
W = 1.0 / (M + 1)  # 1/100 spacing
TIME_LIMIT = 150.0
DEADLINE = time.time() + (TIME_LIMIT - 10.0)  # 10s safety margin


def compute_value(h):
    h = np.asarray(h, dtype=float)
    if np.any(h < -1e-12) or not np.all(np.isfinite(h)):
        return 0.0
    g = np.convolve(h, h)
    y = np.concatenate(([0.0], g, [0.0]))
    l2 = (W / 3.0) * np.sum(y[:-1] ** 2 + y[:-1] * y[1:] + y[1:] ** 2)
    norm_1 = np.sum(np.abs(g)) / (M + 1)
    norm_inf = np.max(np.abs(g))
    if norm_1 <= 0.0 or norm_inf <= 0.0:
        return 0.0
    return l2 / (norm_1 * norm_inf)


def neg_value(x):
    s = np.sum(x)
    if s <= 0.0:
        return 1e10
    h = x / s
    return -compute_value(h)


def optimize_from(x0):
    try:
        res = minimize(neg_value, x0, method="L-BFGS-B",
                       options={"maxiter": 2000, "ftol": 1e-16, "gtol": 1e-12,
                                "maxls": 50})
        h = res.x / np.sum(res.x)
        return compute_value(h), h
    except Exception:
        return 0.0, x0 / max(np.sum(x0), 1e-12)


def plateau(a, b):
    # heights: a at the ends, b in the middle
    h = np.empty(N)
    h[:a] = 1.0
    h[a:N - a] = b
    h[N - a:] = 1.0
    return h


def search_parametric():
    best_v, best_h = 0.0, np.ones(N)
    # coarse grid over plateau family
    for a in range(0, N // 2 + 1):
        for b in np.linspace(0.0, 1.5, 31):
            h = plateau(a, b)
            v = compute_value(h)
            if v > best_v:
                best_v, best_h = v, h.copy()
    # refine around best plateau with L-BFGS-B on full vector
    for _ in range(3):
        v, h = optimize_from(best_h)
        if v > best_v:
            best_v, best_h = v, h
    return best_v, best_h


def search_random(n_restarts):
    rng = np.random.default_rng(0)
    best_v, best_h = 0.0, np.ones(N)
    for r in range(n_restarts):
        if time.time() > DEADLINE:
            break
        # random initial shape: mix of smooth bumps
        t = np.linspace(0, 1, N)
        h = np.zeros(N)
        for _ in range(rng.integers(1, 4)):
            c = rng.uniform(0.2, 0.8)
            sig = rng.uniform(0.05, 0.3)
            h += np.exp(-((t - c) ** 2) / (2 * sig ** 2))
        h += 0.05 * rng.random(N)
        v, h2 = optimize_from(h)
        if v > best_v:
            best_v, best_h = v, h2
    return best_v, best_h


def construct():
    global DEADLINE
    DEADLINE = time.time() + (TIME_LIMIT - 10.0)

    # 1) parametric plateau search (fast, gives a good starting point)
    v1, h1 = search_parametric()
    best_v, best_h = v1, h1.copy()

    # 2) direct L-BFGS-B from the plateau start
    if time.time() < DEADLINE:
        v, h = optimize_from(h1)
        if v > best_v:
            best_v, best_h = v, h.copy()

    # 3) random restarts if time remains
    if time.time() < DEADLINE:
        v, h = search_random(40)
        if v > best_v:
            best_v, best_h = v, h.copy()

    # final polish
    if time.time() < DEADLINE:
        v, h = optimize_from(best_h)
        if v > best_v:
            best_v, best_h = v, h.copy()

    h = np.clip(best_h, 0.0, None)
    if np.sum(h) <= 0:
        h = np.ones(N)
    return [float(x) for x in h]
