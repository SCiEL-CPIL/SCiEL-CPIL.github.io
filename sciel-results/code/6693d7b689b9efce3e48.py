"""Second autocorrelation inequality: optimize 50 nonnegative step heights.

Strategy: Direct multi-start optimization with smooth surrogate for the nonsmooth norm_inf.
The value function is scale-invariant (value = l2/(norm_1*norm_inf)), so only the shape of h matters.
We use a smooth log-sum-exp approximation for max(|g|) during optimization to give L-BFGS-B
well-behaved gradients, then compute the true value at the end.
"""
import time
import numpy as np
from scipy.optimize import minimize

N = 50
M = 2 * N - 1  # 99
W = 1.0 / (M + 1)  # 1/100 spacing
TIME_LIMIT = 150.0
DEADLINE = time.time() + (TIME_LIMIT - 10.0)  # 10s safety margin
_T = 1.0 / 30.0  # Temperature for soft-max approximation


def compute_value(h):
    """Compute the true objective value (non-smooth)."""
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


def compute_value_smooth(h):
    """Compute a smooth approximation of the objective value.
    
    Uses log-sum-exp to approximate max(|g|) smoothly.
    """
    h = np.asarray(h, dtype=float)
    if np.any(h < -1e-12) or not np.all(np.isfinite(h)):
        return 0.0
    g = np.convolve(h, h)
    y = np.concatenate(([0.0], g, [0.0]))
    l2 = (W / 3.0) * np.sum(y[:-1] ** 2 + y[:-1] * y[1:] + y[1:] ** 2)
    norm_1 = np.sum(np.abs(g)) / (M + 1)
    
    # Smooth approximation of max(|g|) using log-sum-exp
    abs_g = np.abs(g)
    max_val = np.max(abs_g)
    if max_val <= 0:
        return 0.0
    # log-sum-exp: max_val + log(sum(exp((abs_g - max_val)/_T))) * _T
    shifted = (abs_g - max_val) / _T
    # Clip for numerical stability
    shifted = np.clip(shifted, -100, 100)
    log_sum_exp = np.log(np.sum(np.exp(shifted)))
    norm_inf_smooth = max_val + _T * log_sum_exp
    
    if norm_1 <= 0.0 or norm_inf_smooth <= 0.0:
        return 0.0
    return l2 / (norm_1 * norm_inf_smooth)


def neg_value(x):
    s = np.sum(x)
    if s <= 0.0:
        return 1e10
    h = x / s
    return -compute_value_smooth(h)


def neg_value_exact(x):
    """Use the exact (non-smooth) value for final polishing."""
    s = np.sum(x)
    if s <= 0.0:
        return 1e10
    h = x / s
    return -compute_value(h)


def optimize_from(x0, use_smooth=True):
    """Optimize from initial point x0.
    
    Args:
        x0: initial point
        use_smooth: if True, use smooth surrogate; if False, use exact value
    """
    try:
        if use_smooth:
            obj = neg_value
        else:
            obj = neg_value_exact
        res = minimize(obj, x0, method="L-BFGS-B",
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


def generate_initialization(rng, t=None):
    """Generate a diverse random initialization."""
    if t is None:
        t = np.linspace(0, 1, N)
    h = np.zeros(N)
    n_bumps = rng.integers(1, 4)
    for _ in range(n_bumps):
        c = rng.uniform(0.2, 0.8)
        sig = rng.uniform(0.05, 0.3)
        h += np.exp(-((t - c) ** 2) / (2 * sig ** 2))
    h += 0.05 * rng.random(N)
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
    # refine around best plateau with L-BFGS-B on full vector using smooth surrogate
    for _ in range(3):
        v, h = optimize_from(best_h, use_smooth=True)
        if v > best_v:
            best_v, best_h = v, h
    return best_v, best_h


def search_random(n_restarts):
    rng = np.random.default_rng(0)
    best_v, best_h = 0.0, np.ones(N)
    t = np.linspace(0, 1, N)
    for r in range(n_restarts):
        if time.time() > DEADLINE:
            break
        h = generate_initialization(rng, t)
        v, h2 = optimize_from(h, use_smooth=True)
        if v > best_v:
            best_v, best_h = v, h2
    return best_v, best_h


def construct():
    global DEADLINE
    DEADLINE = time.time() + (TIME_LIMIT - 10.0)

    # 1) parametric plateau search (fast, gives a good starting point)
    v1, h1 = search_parametric()
    best_v, best_h = v1, h1.copy()

    # 2) direct L-BFGS-B from the plateau start using smooth surrogate
    if time.time() < DEADLINE:
        v, h = optimize_from(h1, use_smooth=True)
        if v > best_v:
            best_v, best_h = v, h.copy()

    # 3) random restarts if time remains
    if time.time() < DEADLINE:
        v, h = search_random(40)
        if v > best_v:
            best_v, best_h = v, h.copy()

    # final polish with exact (non-smooth) value for higher precision
    if time.time() < DEADLINE:
        v, h = optimize_from(best_h, use_smooth=False)
        if v > best_v:
            best_v, best_h = v, h.copy()

    h = np.clip(best_h, 0.0, None)
    if np.sum(h) <= 0:
        h = np.ones(N)
    return [float(x) for x in h]
