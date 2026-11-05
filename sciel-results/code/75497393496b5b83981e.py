"""Optimized program for the third autocorrelation inequality task."""
import numpy as np
import time
from scipy.optimize import minimize

N = 400


def _compute_value(h):
    """Compute the task's objective value exactly as the evaluator does."""
    s = np.sum(h)
    if s == 0:
        return np.inf
    conv = np.convolve(h, h)
    return abs(2 * N * np.max(conv) / (s ** 2))


def _fast_value(h):
    """Fast objective using FFT for autoconvolution (for optimization only)."""
    h = np.asarray(h, dtype=float)
    s = h.sum()
    if abs(s) < 1e-14:
        return 1e12
    # full autoconvolution length 2N-1
    nfft = 1
    while nfft < 2 * N - 1:
        nfft <<= 1
    F = np.fft.rfft(h, nfft)
    conv = np.fft.irfft(F * F, nfft)[:2 * N - 1]
    return abs(2 * N * conv.max() / (s * s))


def construct():
    """Construct step heights minimizing the autocorrelation inequality value."""
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin before 150s limit

    best_h = np.ones(N)
    best_val = _compute_value(best_h)  # 2.0

    def consider(h):
        nonlocal best_h, best_val
        if not np.all(np.isfinite(h)):
            return
        if abs(np.sum(h)) < 1e-12:
            return
        v = _fast_value(h)
        if v < best_val:
            best_val = v
            best_h = h.copy()

    # ---- Phase 1: structured initial patterns ----
    x = np.linspace(-1, 1, N)
    candidates = []
    candidates.append(np.linspace(-1, 1, N))
    candidates.append(np.exp(-2 * x))
    candidates.append(np.exp(2 * x))
    candidates.append(np.sin(np.pi * x) * x)
    candidates.append(np.cos(np.pi * x / 2))
    candidates.append(x ** 2)
    candidates.append(np.sign(x))
    # random structured
    rng = np.random.default_rng(0)
    for _ in range(40):
        h = rng.standard_normal(N)
        h += rng.uniform(-1, 1) * np.linspace(-1, 1, N)
        candidates.append(h)

    for h in candidates:
        h = h - h.mean()
        consider(h)

    # ---- Phase 2: gradient-based refinement from multiple starts ----
    def objective(x):
        return _fast_value(x)

    def grad(x):
        # numerical gradient via FFT-based eval (cheap)
        g = np.zeros(N)
        eps = 1e-6
        base = _fast_value(x)
        for i in range(N):
            xp = x.copy(); xp[i] += eps
            xm = x.copy(); xm[i] -= eps
            g[i] = (_fast_value(xp) - _fast_value(xm)) / (2 * eps)
        return g

    elapsed = time.time() - start_time
    if elapsed < deadline - 60:
        # Refine best found so far with L-BFGS-B (uses numerical grad internally, fast)
        try:
            res = minimize(objective, best_h.copy(), method='L-BFGS-B',
                           options={'maxiter': 400, 'maxfun': 4000, 'maxls': 40,
                                    'ftol': 1e-10, 'gtol': 1e-6})
            if np.isfinite(res.fun) and res.fun < best_val:
                consider(res.x)
        except Exception:
            pass

    # ---- Phase 3: random restarts + short local optimization ----
    elapsed = time.time() - start_time
    while elapsed < deadline - 20:
        h0 = best_h.copy()
        # perturb
        h0 = h0 + 0.3 * rng.standard_normal(N)
        h0 = h0 - h0.mean()
        try:
            res = minimize(objective, h0, method='L-BFGS-B',
                           options={'maxiter': 150, 'maxfun': 1500, 'maxls': 30,
                                    'ftol': 1e-9, 'gtol': 1e-5})
            if np.isfinite(res.fun) and res.fun < best_val:
                consider(res.x)
        except Exception:
            pass
        elapsed = time.time() - start_time

    # ---- Final exact verification and validity guard ----
    if not np.all(np.isfinite(best_h)) or abs(np.sum(best_h)) < 1e-12:
        return np.ones(N)

    # Re-verify with the exact evaluator formula
    final_val = _compute_value(best_h)
    if not np.isfinite(final_val):
        return np.ones(N)

    return best_h
