"""Program for Erdos minimum overlap task using anti-periodic model + progressive smooth optimization.

The objective max(overlap(h,1-h)) is minimized when h is anti-periodic
(h[i+N/2] = 1 - h[i]), which forces the overlap to 0 at the shifts
t = 0, N, 2N (the 'forbidden' shifts). We build a strong anti-periodic
seed and refine it with progressive smooth optimization, sharpening the
surrogate to target the true max correlation.
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 100
HALF = N // 2
DEADLINE = 145.0
START = time.time()


def remaining():
    return DEADLINE - (time.time() - START)


def value(h):
    """The exact evaluator value (lower is better)."""
    c = np.correlate(h, 1.0 - h, mode="full")
    return float(np.max(c)) / N * 2.0


def value_smooth(h, beta):
    """Smooth surrogate for the max via log-sum-exp (lower is better)."""
    c = np.correlate(h, 1.0 - h, mode="full")
    cm = float(c.max())
    z = beta * (c - cm)
    # Numerical stability: clip z to avoid overflow
    z = np.clip(z, -500, 500)
    logsum = np.log(np.sum(np.exp(z))) + beta * cm
    return float(logsum) / N * 2.0


def make_seed():
    """Strong anti-periodic seed: triangular wave, zero-mean per anti-pair."""
    i = np.arange(HALF)
    # triangular wave peaking at the middle of the first half
    d = np.abs(i - (HALF - 1) / 2.0) / ((HALF - 1) / 2.0)
    tri = 1.0 - d
    # center the first half around 0.5 so the anti-periodic completion
    # (1 - first half) stays within [0,1]
    first = 0.5 + (0.5 - tri.mean()) * tri
    h = np.empty(N)
    h[:HALF] = first
    h[HALF:] = 1.0 - first
    # fix tiny numerical drift
    h = np.clip(h, 0.0, 1.0)
    h += (HALF - h.sum()) / N
    h = np.clip(h, 0.0, 1.0)
    return h


def refine_smooth(h, beta, maxiter):
    """Optimize the anti-periodic parameters to lower the smooth objective."""
    h0 = np.asarray(h, dtype=np.float64).copy()
    h0 = np.clip(h0, 1e-12, 1.0 - 1e-12)
    p0 = h0[:HALF].copy()
    bounds = [(1e-12, 1.0 - 1e-12)] * HALF

    def unpack(p):
        p = np.clip(p, 0.0, 1.0)
        hh = np.empty(N)
        hh[:HALF] = p
        hh[HALF:] = 1.0 - p
        return hh

    def obj(p):
        return value_smooth(unpack(p), beta)

    try:
        res = minimize(obj, p0, method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": maxiter, "ftol": 1e-16,
                                "gtol": 1e-14, "maxls": 50})
        return unpack(np.clip(res.x, 0.0, 1.0))
    except Exception:
        return h0


def refine_exact(h, maxiter):
    """Optimize the anti-periodic parameters to lower the exact objective."""
    h0 = np.asarray(h, dtype=np.float64).copy()
    h0 = np.clip(h0, 1e-12, 1.0 - 1e-12)
    p0 = h0[:HALF].copy()
    bounds = [(1e-12, 1.0 - 1e-12)] * HALF

    def unpack(p):
        p = np.clip(p, 0.0, 1.0)
        hh = np.empty(N)
        hh[:HALF] = p
        hh[HALF:] = 1.0 - p
        return hh

    def obj(p):
        return value(unpack(p))

    try:
        res = minimize(obj, p0, method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": maxiter, "ftol": 1e-17,
                                "gtol": 1e-15, "maxls": 60})
        return unpack(np.clip(res.x, 0.0, 1.0))
    except Exception:
        return h0


def greedy_permutation_search(h, max_rounds=500, time_budget=30.0):
    """Greedy local search: swap pairs in the first half to reduce max correlation.
    
    Maintains anti-periodic structure: only permutes values in h[:HALF].
    """
    h = np.asarray(h, dtype=np.float64).copy()
    h = np.clip(h, 1e-12, 1.0 - 1e-12)
    best_val = value(h)
    best_h = h.copy()
    
    t_start = time.time()
    no_improve = 0
    
    for round_idx in range(max_rounds):
        if time.time() - t_start > time_budget:
            break
        if remaining() < 10.0:
            break
        
        improved = False
        # Try all pairs in the first half
        for i in range(HALF):
            if no_improve > 50:
                break
            for j in range(i + 1, HALF):
                # Swap h[i] and h[j] in the first half
                h[i], h[j] = h[j], h[i]
                # Update the anti-periodic second half
                h[HALF + i] = 1.0 - h[i]
                h[HALF + j] = 1.0 - h[j]
                
                v = value(h)
                if v < best_val - 1e-15:
                    best_val = v
                    best_h = h.copy()
                    improved = True
                    no_improve = 0
                else:
                    # Undo swap
                    h[i], h[j] = h[j], h[i]
                    h[HALF + i] = 1.0 - h[i]
                    h[HALF + j] = 1.0 - h[j]
                    no_improve += 1
            
            if not improved and no_improve > 200:
                break
        
        if not improved:
            break
    
    return best_h, best_val


def construct():
    t0 = time.time()

    h = make_seed()
    best = h.copy()
    best_val = value(best)

    # Phase 1: Progressive smooth optimization with increasing beta
    # Start smooth to find good basin, then sharpen to target true max
    beta_schedule = [
        (10.0, 300),
        (20.0, 300),
        (50.0, 300),
        (100.0, 300),
        (200.0, 400),
        (400.0, 500),
        (800.0, 600),
    ]

    for beta, iters in beta_schedule:
        if remaining() < 25.0:
            break
        h = refine_smooth(h, beta, iters)
        v = value(h)
        if v < best_val:
            best, best_val = h.copy(), v

    # Phase 2: Discrete permutation search to escape continuous local minima
    if remaining() > 25.0:
        h_perm, v_perm = greedy_permutation_search(best, max_rounds=500, time_budget=20.0)
        if v_perm < best_val:
            best, best_val = h_perm.copy(), v_perm

    # Phase 3: Exact objective polish with high precision
    if remaining() > 15.0:
        h = refine_exact(best, 500)
        v = value(h)
        if v < best_val:
            best, best_val = h.copy(), v

    # Phase 4: Another round of smooth refinement starting from exact-polished solution
    # This can help escape any local minima introduced by the exact objective's nonsmoothness
    if remaining() > 15.0:
        for beta, iters in [(500.0, 400), (1000.0, 500)]:
            if remaining() < 10.0:
                break
            h = refine_smooth(best, beta, iters)
            v = value(h)
            if v < best_val:
                best, best_val = h.copy(), v

    # Phase 5: Final permutation search from the refined solution
    if remaining() > 15.0:
        h_perm, v_perm = greedy_permutation_search(best, max_rounds=300, time_budget=15.0)
        if v_perm < best_val:
            best, best_val = h_perm.copy(), v_perm

    # Phase 6: Final exact polish
    if remaining() > 10.0:
        h = refine_exact(best, 300)
        v = value(h)
        if v < best_val:
            best, best_val = h.copy(), v

    # Ensure sum constraint is met within tolerance
    total = best.sum()
    if abs(total - HALF) > 1e-9 * N:
        best += (HALF - total) / N
        best = np.clip(best, 0.0, 1.0)

    return best
