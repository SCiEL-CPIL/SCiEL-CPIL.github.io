"""Optimize the shape of 50 nonnegative heights to maximize
   ||h||_2^2 / (||h||_1 * ||h*h||_inf), which is scale-invariant.
   Uses a smooth objective with multiple restarts + local refinement,
   guarded by an explicit wall-clock deadline."""

import time
import math
import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 50
# wall-clock budget: evaluator kills at 150s; keep a safety margin.
_DEADLINE = 140.0
_START = time.time()

def _time_left():
    return _DEADLINE - (time.time() - _START)

def _autoconv_peak(h):
    """Supremum of the discrete autoconvolution of nonnegative h.
    For h >= 0 the maximum is at lag 0: sum_i h_i^2.
    We compute the full circular-free linear convolution peak to be safe."""
    # linear autoconvolution
    c = np.convolve(h, h, mode='full')
    return float(c.max())

def _ratio(h):
    h = np.asarray(h, dtype=float)
    h = np.clip(h, 0.0, None)
    l1 = h.sum()
    if l1 <= 0:
        return 0.0
    l2sq = float(h @ h)
    peak = _autoconv_peak(h)
    if peak <= 0:
        return 0.0
    return l2sq / (l1 * peak)

def _smooth_ratio(h):
    """Smooth surrogate: replace the hard max in the autoconvolution peak
    with a softmax over the autoconvolution values to get a differentiable
    objective. Scale-invariant, so we normalize h to unit L1 inside."""
    h = np.clip(np.asarray(h, dtype=float), 0.0, None)
    l1 = h.sum()
    if l1 <= 0:
        return 0.0
    h = h / l1  # unit L1 -> scale-free
    l2sq = float(h @ h)
    c = np.convolve(h, h, mode='full')
    # softmax peak
    m = c.max()
    z = c - m
    # numerical stability
    z = np.clip(z, -50.0, 0.0)
    peak = m + float(np.log(np.sum(np.exp(z))))
    if peak <= 0:
        return 0.0
    return l2sq / peak  # l1 == 1

def _neg_smooth(h):
    return -_smooth_ratio(h)

def _one_sided_profile(alpha, beta, n=N):
    """One-sided power/exponential profile concentrated on one side.
    alpha controls decay, beta controls tilt. Returns nonnegative array."""
    idx = np.arange(n, dtype=float)
    # decay from one end
    t = idx / max(n - 1, 1)
    h = np.exp(-alpha * t) * (1.0 + beta * t)
    h = np.clip(h, 0.0, None)
    if h.sum() <= 0:
        h = np.ones(n)
    return h

def _random_spread(rng):
    """Random nonnegative shape with spread mass (smooth-ish)."""
    # random smooth profile via few random sinusoids
    a = rng.uniform(0.5, 3.0, size=3)
    b = rng.uniform(0.0, 2.0, size=3)
    x = np.linspace(0.0, 1.0, N)
    h = np.ones(N)
    for k in range(3):
        h = h * (1.0 + 0.5 * np.sin(a[k] * x + b[k]))
    h = np.clip(h, 0.0, None)
    if h.sum() <= 0:
        h = np.ones(N)
    return h

def _local_refine(h0, max_iter=400):
    """Direct coordinate/ascent refinement on the exact ratio.
    Scale-invariant, so we normalize to unit L1 and move one coordinate at a time."""
    h = np.clip(np.asarray(h0, dtype=float), 0.0, None)
    if h.sum() <= 0:
        h = np.ones(N)
    h = h / h.sum()
    best = _ratio(h)
    for _ in range(max_iter):
        if _time_left() < 2.0:
            break
        improved = False
        # try small multiplicative perturbations on each coordinate
        for i in range(N):
            if _time_left() < 2.0:
                break
            base = h[i]
            for factor in (0.5, 0.75, 1.5, 2.0, 0.0):
                h[i] = base * factor
                r = _ratio(h)
                if r > best + 1e-12:
                    best = r
                    improved = True
                else:
                    h[i] = base
        if not improved:
            break
    return h, best

def construct():
    global _START, _DEADLINE
    _START = time.time()
    _DEADLINE = _START + 140.0

    rng = np.random.default_rng(12345)
    best_h = None
    best_val = -1.0

    def consider(h):
        nonlocal best_h, best_val
        h = np.clip(np.asarray(h, dtype=float), 0.0, None)
        if h.sum() <= 0:
            return
        v = _ratio(h)
        if v > best_val:
            best_val = v
            best_h = h.copy()

    # ---- Phase 1: analytic one-sided profiles (fast, strong seed) ----
    for alpha in [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0]:
        for beta in [0.0, 0.5, 1.0, 2.0, 3.0]:
            h = _one_sided_profile(alpha, beta)
            consider(h)
            # also mirror
            consider(h[::-1].copy())

    # uniform baseline
    consider(np.ones(N))

    # ---- Phase 2: random smooth shapes ----
    for _ in range(40):
        if _time_left() < 20.0:
            break
        h = _random_spread(rng)
        consider(h)

    # ---- Phase 3: global-ish optimization over analytic params ----
    # Optimize (alpha, beta) of the one-sided profile via the smooth objective
    if _time_left() > 30.0:
        def obj(x):
            a, b = x
            h = _one_sided_profile(a, b)
            return _neg_smooth(h)
        best_res = None
        best_f = np.inf
        for start in [(1.0, 1.0), (2.0, 0.5), (3.0, 1.5), (4.0, 2.0), (0.7, 0.3)]:
            if _time_left() < 15.0:
                break
            try:
                res = minimize(obj, np.array(start, dtype=float),
                               method='Nelder-Mead',
                               options={'maxiter': 300, 'xatol': 1e-6, 'fatol': 1e-8})
                if res.fun < best_f:
                    best_f = res.fun
                    best_res = res
            except Exception:
                pass
        if best_res is not None:
            a, b = best_res.x
            h = _one_sided_profile(float(a), float(b))
            consider(h)
            consider(h[::-1].copy())

    # ---- Phase 4: local refinement from the best shapes ----
    if _time_left() > 10.0 and best_h is not None:
        # refine the current best
        h_ref, v_ref = _local_refine(best_h, max_iter=300)
        if v_ref > best_val:
            best_val = v_ref
            best_h = h_ref

        # refine a couple of other strong candidates
        cands = []
        for alpha in [1.0, 2.0, 3.0]:
            for beta in [0.5, 1.0, 2.0]:
                cands.append(_one_sided_profile(alpha, beta))
                cands.append(_one_sided_profile(alpha, beta)[::-1].copy())
        for c in cands[:8]:
            if _time_left() < 8.0:
                break
            h_ref, v_ref = _local_refine(c, max_iter=200)
            if v_ref > best_val:
                best_val = v_ref
                best_h = h_ref

    # ---- Final: one more refinement pass on the overall best if time allows ----
    if _time_left() > 5.0 and best_h is not None:
        h_ref, v_ref = _local_refine(best_h, max_iter=400)
        if v_ref > best_val:
            best_val = v_ref
            best_h = h_ref

    if best_h is None:
        return [1.0] * N

    # The ratio is scale-invariant; return the shape. Keep values nonnegative.
    # Normalize to a reasonable magnitude (does not affect the scale-free ratio).
    h_out = np.clip(best_h, 0.0, None)
    if h_out.sum() <= 0:
        h_out = np.ones(N)
    return [float(x) for x in h_out]
