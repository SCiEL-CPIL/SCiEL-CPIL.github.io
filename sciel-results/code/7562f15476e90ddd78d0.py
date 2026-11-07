"""Fresh design: parametric-shape global search + full-dim local polish for the
second autocorrelation inequality value. Entry point: construct() -> 50 heights."""
import time
import numpy as np
from scipy.optimize import minimize

N = 50
N_STARTS = 60          # random parametric starts
POLISH_STARTS = 8      # local polish attempts on full 50-dim
DEADLINE = 140.0       # safety margin under 150s wall-clock

# ---- fast evaluator (mirrors the contract exactly) ----
def _value_from_h(h):
    h = np.asarray(h, dtype=float)
    g = np.convolve(h, h)
    m = len(g)
    y = np.concatenate(([0.0], g, [0.0]))
    w = 1.0 / (m + 1)
    # l2 = sum_i (w/3) * (y_i^2 + y_i y_{i+1} + y_{i+1}^2)  for i in 0..m
    a = y[:-1]
    b = y[1:]
    l2 = (w / 3.0) * np.sum(a * a + a * b + b * b)
    norm_1 = np.sum(np.abs(g)) / (m + 1)
    norm_inf = np.max(np.abs(g))
    if norm_1 <= 0 or norm_inf <= 0:
        return 0.0
    return l2 / (norm_1 * norm_inf)

# ---- parametric shape model: 12-dim -> 50 nonnegative heights ----
def _params_to_h(p):
    # p = [base, width, exp, center, asym, peak, taper, skew, bump, bumpw,
    #      bumpscale, jitter_amp]
    base = p[0]
    width = p[1]
    exp = p[2]
    center = p[3]
    asym = p[4]
    peak = p[5]
    taper = p[6]
    skew = p[7]
    bump = p[8]
    bumpw = p[9]
    bumpscale = p[10]
    jitter_amp = p[11]

    x = np.linspace(-1.0, 1.0, N)
    # main bell
    bell = np.exp(-((x - center) ** 2) / (2.0 * max(width, 1e-3) ** 2))
    # exponent shaping (make it sharper/plateau)
    bell = np.power(bell, max(exp, 0.05))
    # asymmetry: modulate left vs right
    bell = bell * (1.0 + asym * x)
    # skew: shift effective center with a linear term
    bell = bell * (1.0 + skew * x * x)
    # secondary bump
    bump_term = bumpscale * np.exp(-((x - bump) ** 2) / (2.0 * max(bumpw, 1e-3) ** 2))
    # taper the ends
    taper_term = 1.0 + taper * (np.abs(x) ** 3)
    h = base + peak * bell + bump_term
    h = h * taper_term
    # tiny deterministic jitter to break exact symmetry (avoid plateaus)
    h = h + jitter_amp * np.sin(2.0 * np.pi * x * 1.7 + 0.3)
    h = np.clip(h, 0.0, None)
    # normalize scale (value is scale-invariant? check: g scales as h^2, l2 as h^4,
    # norm_1*norm_inf as h^4 -> scale invariant, so scale is free; keep it bounded)
    mx = np.max(h)
    if mx > 0:
        h = h * (1.0 / mx)   # normalize peak to 1 (scale-invariant, keeps numbers stable)
    return h

def _neg_value_params(p):
    try:
        h = _params_to_h(p)
        if not np.all(np.isfinite(h)) or np.any(h < 0):
            return 1e6
        return -_value_from_h(h)
    except Exception:
        return 1e6

def _neg_value_full(h):
    h = np.clip(np.asarray(h, dtype=float), 0.0, None)
    try:
        if not np.all(np.isfinite(h)):
            return 1e6
        return -_value_from_h(h)
    except Exception:
        return 1e6

# ---- search ----
def _global_search(start_time, best_h, best_val):
    rng = np.random.default_rng(12345)
    # bounds for the 12 params
    lb = np.array([0.0, 0.05, 0.1, -0.6, -0.8, 0.0, -0.5, -0.5, -0.8, 0.05, -1.0, 0.0])
    ub = np.array([1.0, 1.5, 4.0, 0.6, 0.8, 2.0, 0.5, 0.5, 0.8, 1.5, 1.0, 0.05])
    for s in range(N_STARTS):
        if time.time() - start_time > DEADLINE:
            break
        p0 = lb + (ub - lb) * rng.random(12)
        # bias toward a good bell: center near 0, moderate width
        p0[3] = (p0[3] * 0.3)  # center mostly near 0
        try:
            res = minimize(_neg_value_params, p0, method='L-BFGS-B',
                           bounds=list(zip(lb, ub)),
                           options={'maxiter': 400, 'ftol': 1e-12, 'gtol': 1e-8})
        except Exception:
            continue
        if res.success or res.fun < 1e5:
            p = res.x
            h = _params_to_h(p)
            v = _value_from_h(h)
            if v > best_val:
                best_val = v
                best_h = h.copy()
    return best_h, best_val

def _local_polish(start_time, h0, max_val):
    # polish on full 50-dim from a few perturbations of the best parametric result
    rng = np.random.default_rng(999)
    best_h = h0.copy()
    best_val = max_val
    for k in range(POLISH_STARTS):
        if time.time() - start_time > DEADLINE:
            break
        if k == 0:
            h = h0.copy()
        else:
            h = h0 + rng.normal(0.0, 0.05, N)
            h = np.clip(h, 0.0, None)
        try:
            res = minimize(_neg_value_full, h, method='L-BFGS-B',
                           bounds=[(0.0, None)] * N,
                           options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-7})
        except Exception:
            continue
        h2 = np.clip(res.x, 0.0, None)
        v = _value_from_h(h2)
        if v > best_val:
            best_val = v
            best_h = h2.copy()
    return best_h, best_val

def construct():
    start_time = time.time()
    # seed baseline
    seed_h = np.ones(N)
    best_h = seed_h.copy()
    best_val = _value_from_h(seed_h)

    # global parametric search
    best_h, best_val = _global_search(start_time, best_h, best_val)

    # local polish on full dimension
    best_h, best_val = _local_polish(start_time, best_h, best_val)

    # final safety: ensure valid, finite, nonnegative
    best_h = np.asarray(best_h, dtype=float)
    best_h = np.clip(best_h, 0.0, None)
    if not np.all(np.isfinite(best_h)):
        best_h = np.ones(N)
    return list(best_h)
