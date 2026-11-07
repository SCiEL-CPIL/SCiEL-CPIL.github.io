"""Program for Erdos minimum overlap task using anti-periodic model + discrete permutation search.

The objective max(overlap(h,1-h)) is minimized when h is anti-periodic
(h[i+N/2] = 1 - h[i]), which forces the overlap to 0 at the shifts
t = 0, N, 2N (the 'forbidden' shifts). We first build a strong anti-periodic
seed via continuous optimization, then refine via discrete permutation search
on a symmetric grid of values, using greedy swaps to minimize the max correlation.
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 100
HALF = N // 2
DEADLINE = 147.0
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
    logsum = np.log(np.sum(np.exp(z))) + beta * cm
    return float(logsum) / N * 2.0


def make_seed():
    """Strong anti-periodic seed: triangular wave, zero-mean per anti-pair."""
    i = np.arange(HALF)
    d = np.abs(i - (HALF - 1) / 2.0) / ((HALF - 1) / 2.0)
    tri = 1.0 - d
    first = 0.5 + (0.5 - tri.mean()) * tri
    h = np.empty(N)
    h[:HALF] = first
    h[HALF:] = 1.0 - first
    h = np.clip(h, 0.0, 1.0)
    h += (HALF - h.sum()) / N
    h = np.clip(h, 0.0, 1.0)
    return h


def refine_continuous(h, beta, maxiter):
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
                       options={"maxiter": maxiter, "ftol": 1e-15,
                                "gtol": 1e-12, "maxls": 40})
        return unpack(np.clip(res.x, 0.0, 1.0))
    except Exception:
        return h0


def snap_to_grid(h, n_levels):
    """Snap h to a symmetric grid of n_levels values, preserving anti-periodicity.
    
    We quantize the first half to a grid, then set second half = 1 - first.
    The grid is chosen so that the multiset is symmetric: if v is in the grid,
    so is 1-v. We use levels at {0, 1/(n-1), 2/(n-1), ..., 1} but only take
    the first half's values from the lower portion to keep them in [0,1].
    
    Actually, for anti-periodicity, we need h[i+HALF] = 1 - h[i]. So if we
    quantize h[:HALF] to some values, h[HALF:] is determined. The sum constraint
    is automatically satisfied if the mean of h[:HALF] is 0.5.
    
    We'll quantize h[:HALF] to a grid and adjust to satisfy sum constraint.
    """
    h = np.asarray(h, dtype=np.float64)
    first = h[:HALF].copy()
    
    # Quantize to grid: use n_levels/2 levels for the first half, centered around 0.5
    # Grid points for first half: we want values in [0,1] with mean ~0.5
    # Use a symmetric grid: {0, 1/(n-1), ..., 1} but for first half, use all n_levels
    # However, to maintain sum=HALF, we need mean(first)=0.5
    
    # Simple approach: quantize to {0, 0.5, 1} if n_levels=3, or finer grid
    # Let's use n_levels grid points evenly spaced in [0,1]
    if n_levels <= 1:
        first_q = np.full(HALF, 0.5)
    else:
        grid = np.linspace(0.0, 1.0, n_levels)
        # Find nearest grid point for each value in first
        idx = np.argmin(np.abs(first[:, None] - grid[None, :]), axis=1)
        first_q = grid[idx]
    
    # Adjust to satisfy sum constraint: sum(first_q) should be HALF
    # We can do this by adjusting some elements slightly
    target_sum = HALF
    current_sum = first_q.sum()
    diff = target_sum - current_sum
    
    if abs(diff) > 1e-10:
        # Find elements that can be adjusted without violating [0,1]
        # Simple approach: distribute the difference evenly
        first_q += diff / HALF
        first_q = np.clip(first_q, 0.0, 1.0)
        # Re-clip and re-adjust if needed (iterative)
        for _ in range(10):
            diff = target_sum - first_q.sum()
            if abs(diff) < 1e-12:
                break
            first_q += diff / HALF
            first_q = np.clip(first_q, 0.0, 1.0)
    
    h_out = np.empty(N)
    h_out[:HALF] = first_q
    h_out[HALF:] = 1.0 - first_q
    h_out = np.clip(h_out, 0.0, 1.0)
    
    # Final sum check
    total = h_out.sum()
    if abs(total - HALF) > 1e-9 * N:
        h_out += (HALF - total) / N
        h_out = np.clip(h_out, 0.0, 1.0)
    
    return h_out


def greedy_permutation_search(h, max_iters=5000):
    """Greedy local search: swap pairs in the first half to reduce max correlation.
    
    We fix the multiset of values (anti-periodic structure) and only permute
    the order of values in the first half. Each swap is accepted if it reduces
    the exact objective value.
    """
    h = np.asarray(h, dtype=np.float64).copy()
    first = h[:HALF].copy()
    best_val = value(h)
    
    # Precompute: we'll evaluate value for candidate swaps
    # For efficiency, we can compute the correlation and find the max
    # But for simplicity, we'll just recompute value each time
    
    # Use a randomized order for swaps to escape local minima
    rng = np.random.default_rng(42)
    
    improved = True
    iters = 0
    while improved and iters < max_iters and remaining() > 20.0:
        improved = False
        iters += 1
        # Try all pairs of positions in the first half
        for i in range(HALF):
            if remaining() < 20.0:
                break
            for j in range(i + 1, HALF):
                if remaining() < 20.0:
                    break
                # Swap first[i] and first[j]
                first[i], first[j] = first[j], first[i]
                h_candidate = np.empty(N)
                h_candidate[:HALF] = first
                h_candidate[HALF:] = 1.0 - first
                v_candidate = value(h_candidate)
                
                if v_candidate < best_val - 1e-12:
                    best_val = v_candidate
                    improved = True
                else:
                    # Revert
                    first[i], first[j] = first[j], first[i]
    
    h_out = np.empty(N)
    h_out[:HALF] = first
    h_out[HALF:] = 1.0 - first
    return h_out, best_val


def simulated_annealing_permutation(h, max_iters=20000):
    """Simulated annealing on permutations of the first half.
    
    Accepts worse moves with decreasing probability to escape local minima.
    """
    h = np.asarray(h, dtype=np.float64).copy()
    first = h[:HALF].copy()
    best_first = first.copy()
    best_val = value(h)
    current_val = best_val
    
    rng = np.random.default_rng(123)
    
    T0 = 0.1
    T_end = 1e-6
    n_steps = min(max_iters, int(remaining() * 100))  # ~100 steps per second
    if n_steps < 100:
        n_steps = 100
    
    for it in range(n_steps):
        if remaining() < 25.0:
            break
        T = T0 * (T_end / T0) ** (it / max(1, n_steps - 1))
        
        # Pick two random positions to swap
        i = rng.integers(0, HALF)
        j = rng.integers(0, HALF)
        if i == j:
            continue
        
        first[i], first[j] = first[j], first[i]
        h_candidate = np.empty(N)
        h_candidate[:HALF] = first
        h_candidate[HALF:] = 1.0 - first
        v_candidate = value(h_candidate)
        
        delta = v_candidate - current_val
        if delta < 0 or rng.random() < np.exp(-delta / max(T, 1e-10)):
            current_val = v_candidate
            if v_candidate < best_val:
                best_val = v_candidate
                best_first = first.copy()
        else:
            # Revert
            first[i], first[j] = first[j], first[i]
    
    h_out = np.empty(N)
    h_out[:HALF] = best_first
    h_out[HALF:] = 1.0 - best_first
    return h_out, best_val


def construct():
    t0 = time.time()

    # Phase 1: Continuous optimization to get a good anti-periodic seed
    h = make_seed()
    best = h.copy()
    best_val = value(best)

    for beta, iters in [(20.0, 150), (50.0, 150), (120.0, 150)]:
        if remaining() < 40.0:
            break
        h = refine_continuous(h, beta, iters)
        v = value(h)
        if v < best_val:
            best, best_val = h.copy(), v

    # Phase 2: Snap to discrete grid and do permutation search
    # Try different grid resolutions
    for n_levels in [3, 5, 7, 11]:
        if remaining() < 30.0:
            break
        
        h_grid = snap_to_grid(best, n_levels)
        v_grid = value(h_grid)
        
        # Greedy permutation search
        if remaining() > 25.0:
            h_greedy, v_greedy = greedy_permutation_search(h_grid, max_iters=3000)
            if v_greedy < best_val:
                best, best_val = h_greedy.copy(), v_greedy
        
        # Simulated annealing (if time permits)
        if remaining() > 25.0:
            h_sa, v_sa = simulated_annealing_permutation(h_greedy if 'h_greedy' in locals() else h_grid, max_iters=10000)
            if v_sa < best_val:
                best, best_val = h_sa.copy(), v_sa

    # Phase 3: Final continuous polish on the best discrete solution
    if remaining() > 15.0:
        p0 = np.clip(best[:HALF], 0.0, 1.0).copy()
        bounds = [(0.0, 1.0)] * HALF

        def unpack(p):
            p = np.clip(p, 0.0, 1.0)
            hh = np.empty(N)
            hh[:HALF] = p
            hh[HALF:] = 1.0 - p
            return hh

        try:
            res = minimize(lambda p: value(unpack(p)), p0, method="L-BFGS-B",
                           bounds=bounds,
                           options={"maxiter": 200, "ftol": 1e-16,
                                    "gtol": 1e-13, "maxls": 40})
            h = unpack(np.clip(res.x, 0.0, 1.0))
            v = value(h)
            if v < best_val:
                best, best_val = h.copy(), v
        except Exception:
            pass

    return best
