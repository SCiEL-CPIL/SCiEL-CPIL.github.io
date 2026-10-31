import time
import math
import numpy as np

N = 600

# ---------------------------------------------------------------------------
# FFT-based linear autoconvolution and its gradient (all vectorized).
# ---------------------------------------------------------------------------
def _autoconv(h):
    """Linear autoconvolution of the step heights (length 2N-1)."""
    L = 2 * N - 1
    f = np.fft.rfft(h, n=L)
    return np.fft.irfft(f * f, n=L)

def _autoconv_grad(h, c):
    """Gradient of sum(c) w.r.t. h, i.e. 2 * (h * h) linearly convolved."""
    L = 2 * N - 1
    fh = np.fft.rfft(h, n=L)
    fc = np.fft.rfft(c, n=L)
    return 2.0 * np.fft.irfft(fh * fc, n=L)[:N]

def _lsq_sum(c, tau):
    """Smooth log-sum-exp surrogate: tau * log(sum(exp(c/tau)))."""
    m = c.max()
    return tau * math.log(float(np.sum(np.exp((c - m) / tau)))) + m

def _lsq_sum_grad(c, tau):
    """Gradient of the log-sum-exp surrogate w.r.t. c (softmax)."""
    m = c.max()
    e = np.exp((c - m) / tau)
    return e / float(e.sum())

def _score(h):
    return float(_autoconv(h).max())

def _project_nonneg(h):
    return np.maximum(h, 0.0)

def _project_nonneg_sum1(h):
    h = np.maximum(h, 0.0)
    s = h.sum()
    if s <= 0.0:
        return np.full(N, 1.0 / N)
    return h / s

def _run_lsq(h, tau, iters, budget, best_h, best_s):
    """Projected gradient descent on the log-sum-exp surrogate."""
    h = h.copy()
    for _ in range(iters):
        if time.time() >= budget:
            break
        c = _autoconv(h)
        g = _autoconv_grad(h, _lsq_sum_grad(c, tau))
        # backtracking line search on the smooth surrogate
        t = 1.0
        f0 = _lsq_sum(c, tau)
        for _ in range(30):
            h2 = _project_nonneg(h - t * g)
            c2 = _autoconv(h2)
            f2 = _lsq_sum(c2, tau)
            if f2 <= f0 - 1e-4 * t * float(np.dot(g, g)):
                h = h2
                break
            t *= 0.5
        else:
            h = _project_nonneg(h - t * g)
        if time.time() >= budget:
            break
        s = _score(h)
        if s < best_s:
            best_s = s
            best_h = h.copy()
    return best_h, best_s

def _run_lsq_norm(h, tau, iters, budget, best_h, best_s):
    """Projected gradient descent with a fixed-step normalization constraint."""
    h = h.copy()
    for _ in range(iters):
        if time.time() >= budget:
            break
        c = _autoconv(h)
        g = _autoconv_grad(h, _lsq_sum_grad(c, tau))
        # backtracking line search (surrogate) on the normalized manifold
        t = 1.0
        f0 = _lsq_sum(c, tau)
        for _ in range(30):
            h2 = _project_nonneg_sum1(h - t * g)
            c2 = _autoconv(h2)
            f2 = _lsq_sum(c2, tau)
            if f2 <= f0 - 1e-4 * t * float(np.dot(g, g)):
                h = h2
                break
            t *= 0.5
        else:
            h = _project_nonneg_sum1(h - t * g)
        if time.time() >= budget:
            break
        s = _score(h)
        if s < best_s:
            best_s = s
            best_h = h.copy()
    return best_h, best_s

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def construct():
    t0 = time.time()
    # generous internal deadline so the whole construct() finishes well
    # before the evaluator's wall-clock limit.
    budget = t0 + 130.0

    best_h = np.ones(N)
    best_s = _score(best_h)

    # ---- Stage 1: unconstrained nonnegativity, multiple tau schedules ----
    for tau0, iters in [(0.15, 150), (0.05, 250)]:
        if time.time() >= budget:
            break
        h = best_h.copy()
        # anneal tau down for sharper peak control
        for frac in (1.0, 0.5, 0.25):
            tau = tau0 * frac
            rem = budget - time.time()
            if rem < 0.5:
                break
            step_budget = budget - 1.0
            best_h, best_s = _run_lsq(h, tau, iters, step_budget, best_h, best_s)
            h = best_h.copy()

    # ---- Stage 2: normalization-constrained refinement (scale-free) ----
    if time.time() < budget - 1.0:
        h = _project_nonneg_sum1(best_h)
        for tau in (0.03, 0.01):
            if time.time() >= budget - 1.0:
                break
            best_h, best_s = _run_lsq_norm(h, tau, 200, budget - 1.0, best_h, best_s)
            h = best_h.copy()

    # ---- Stage 3: final sharpening on the unconstrained problem ----
    if time.time() < budget - 1.0:
        h = best_h.copy()
        best_h, best_s = _run_lsq(h, 0.005, 300, budget - 1.0, best_h, best_s)

    # Safety margin: always return before the hard limit.
    return [float(x) for x in best_h]
