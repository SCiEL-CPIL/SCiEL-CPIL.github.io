"""Circle packing in perimeter-4 rectangle: Differential Evolution + Max-Radius Feasibility."""
import numpy as np
import time
from scipy.optimize import differential_evolution

N = 21


def _feasibility(x, y, w, h, r_scale):
    """Check if circles with radii r_scale*r_base fit. Returns True if valid."""
    # Wall constraints
    if np.any(x - r_scale * r_base < -1e-10) or np.any(y - r_scale * r_base < -1e-10):
        return False
    if np.any(x + r_scale * r_base > w + 1e-10) or np.any(y + r_scale * r_base > h + 1e-10):
        return False
    # Pairwise
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d_sq = dx * dx + dy * dy
    r_sum = r_scale * (r_base[:, None] + r_base[None, :])
    iu = np.triu_indices(N, k=1)
    if np.any(d_sq[iu] < r_sum[iu] ** 2 - 1e-10):
        return False
    return True


def _max_scale(x, y, w, h, r_base, lo=0.0, hi=2.0, iters=40):
    """Binary search for max uniform scale factor."""
    # Quick upper bound from walls
    for i in range(N):
        max_r_wall = min(x[i], w - x[i], y[i], h - y[i])
        if r_base[i] > 1e-12:
            ub = max_r_wall / r_base[i]
            if ub < hi:
                hi = ub
    if hi <= lo:
        return lo
    for _ in range(iters):
        mid = (lo + hi) / 2.0
        if _feasibility(x, y, w, h, mid * np.ones(N) * 0 + mid) if False else _feasibility_at_scale(x, y, w, h, r_base, mid):
            lo = mid
        else:
            hi = mid
    return lo


def _feasibility_at_scale(x, y, w, h, r_base, s):
    """Check if circles with radii s*r_base fit in rectangle w x h."""
    r = s * r_base
    if np.any(r < 0):
        return False
    if np.any(x - r < -1e-10) or np.any(y - r < -1e-10):
        return False
    if np.any(x + r > w + 1e-10) or np.any(y + r > h + 1e-10):
        return False
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d_sq = dx * dx + dy * dy
    r_sum = r[:, None] + r[None, :]
    iu = np.triu_indices(N, k=1)
    if np.any(d_sq[iu] < r_sum[iu] ** 2 - 1e-10):
        return False
    return True


def _objective(params):
    """
    params: [w, x_0..x_N-1, y_0..y_N-1, r_base_0..r_base_N-1]
    Total: 1 + 2*N + N = 3*N + 1 = 64
    Objective: -max_scale (to maximize)
    """
    w = params[0]
    h = 2.0 - w
    if w < 0.3 or h < 0.3:
        return 1e6
    x = params[1:1 + N]
    y = params[1 + N:1 + 2 * N]
    r_base = params[1 + 2 * N:1 + 3 * N]

    # Normalize r_base to sum=1 for better conditioning
    r_sum_base = np.sum(r_base)
    if r_sum_base < 1e-6:
        return 1e6
    r_norm = r_base / r_sum_base

    # Max scale: we want sum(r) = s * sum(r_norm) = s, so maximizing s = maximizing sum(r)
    # Actually r_i = s * r_norm_i, sum(r) = s * sum(r_norm) = s * 1 = s
    # So sum of radii = s. We maximize s.
    s = _max_scale_vec(x, y, w, h, r_norm)
    return -s


def _max_scale_vec(x, y, w, h, r_norm):
    """Vectorized binary search for max scale s such that circles s*r_norm fit."""
    # Upper bound from walls
    hi = np.inf
    for i in range(N):
        if r_norm[i] > 1e-12:
            ub = min(x[i], w - x[i], y[i], h - y[i]) / r_norm[i]
            if ub < hi:
                hi = ub
    if not np.isfinite(hi):
        hi = 1.0
    lo = 0.0
    if hi < 1e-8:
        return 0.0
    # Binary search
    for _ in range(35):
        mid = (lo + hi) / 2.0
        if _feasibility_at_scale(x, y, w, h, r_norm, mid):
            lo = mid
        else:
            hi = mid
    return lo


def _is_valid_packing(x, y, r, w, h, tol=1e-8):
    """Full validity check."""
    if w <= 0.05 or h <= 0.05:
        return False
    if np.any(r < 0):
        return False
    if np.any(x - r < -tol) or np.any(y - r < -tol):
        return False
    if np.any(x + r > w + tol) or np.any(y + r > h + tol):
        return False
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d_sq = dx * dx + dy * dy
    r_sum = r[:, None] + r[None, :]
    iu = np.triu_indices(N, k=1)
    if np.any(d_sq[iu] < r_sum[iu] ** 2 - tol):
        return False
    return True


def _polish_positions(x, y, r, w, h, time_budget):
    """
    Given valid packing, try to improve by:
    1. Force-directed relaxation (positions only)
    2. Per-circle radius expansion with binary search
    3. Uniform scale-up
    Returns improved (x, y, r, w, h) or None.
    """
    n = N
    x = x.copy()
    y = y.copy()
    r = r.copy()

    # Phase 1: Force-directed position relaxation
    dt = 0.002
    for it in range(2000):
        if time.time() > time_budget:
            break
        fx = np.zeros(n)
        fy = np.zeros(n)

        # Wall forces
        fx += np.maximum(r - x, 0) * 50
        fx -= np.maximum(x + r - w, 0) * 50
        fy += np.maximum(r - y, 0) * 50
        fy -= np.maximum(y + r - h, 0) * 50

        # Pairwise repulsion
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d_sq = dx * dx + dy * dy
        r_sum = r[:, None] + r[None, :]
        iu = np.triu_indices(n, k=1)
        d_pairs = np.sqrt(np.maximum(d_sq[iu], 1e-12))
        overlap = np.maximum(r_sum[iu] - d_pairs, 0)

        if np.max(overlap) < 1e-10:
            break

        f_pairs = overlap * 20.0
        dx_p = dx[iu]
        dy_p = dy[iu]
        ux = dx_p / d_pairs
        uy = dy_p / d_pairs

        for idx in range(len(iu[0])):
            ii = iu[0][idx]
            jj = iu[1][idx]
            fx[ii] += ux[idx] * f_pairs[idx]
            fy[ii] += uy[idx] * f_pairs[idx]
            fx[jj] -= ux[idx] * f_pairs[idx]
            fy[jj] -= uy[idx] * f_pairs[idx]

        x += dt * fx
        y += dt * fy
        x = np.clip(x, r, w - r)
        y = np.clip(y, r, h - r)

    # Phase 2: Per-circle radius expansion
    for rnd in range(20):
        if time.time() > time_budget:
            break
        improved = False
        for i in range(n):
            r_wall = min(x[i], w - x[i], y[i], h - y[i])
            r_max = r_wall
            for j in range(n):
                if i == j:
                    continue
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx * dx + dy * dy)
                r_allowed = d - r[j]
                if r_allowed < r_max:
                    r_max = r_allowed
            r_max = max(0.0, r_max)
            if r_max > r[i] + 1e-7:
                lo, hi = r[i], r_max
                for _ in range(25):
                    mid = (lo + hi) / 2.0
                    r_trial = r.copy()
                    r_trial[i] = mid
                    if _is_valid_packing(x, y, r_trial, w, h, tol=1e-9):
                        lo = mid
                    else:
                        hi = mid
                if lo > r[i] + 1e-8:
                    r[i] = lo
                    improved = True
        if not improved:
            break

    # Phase 3: Uniform scale-up
    for _ in range(100):
        if time.time() > time_budget:
            break
        r_trial = r * 1.0005
        if _is_valid_packing(x, y, r_trial, w, h, tol=1e-9):
            r = r_trial
        else:
            break

    if _is_valid_packing(x, y, r, w, h):
        return x, y, r, w, h
    return None


def _try_aspect_ratio(w, x, y, r_base, time_budget):
    """Try a specific aspect ratio, optimize positions via DE, then polish."""
    h = 2.0 - w
    if h < 0.3:
        return None

    # Scale r_base to be reasonable for this rectangle
    # Target: sum of radii should be around area / (pi/4) * packing_efficiency
    area = w * h
    # Rough: for N circles, avg radius ~ sqrt(area/N) * 0.6
    avg_r = np.sqrt(area / N) * 0.55
    r_norm = r_base / np.sum(r_base)
    r_init = r_norm * avg_r

    # DE over positions only (radii fixed at r_init, then we scale up)
    bounds = [(r_init[i], w - r_init[i]) for i in range(N)] + \
             [(r_init[i], h - r_init[i]) for i in range(N)]

    def pos_objective(params):
        x = params[:N]
        y = params[N:2 * N]
        # Check if r_init fits
        if not _is_valid_packing(x, y, r_init, w, h, tol=1e-6):
            # Penalty: sum of violations
            pen = 0.0
            pen += np.sum(np.maximum(r_init - x, 0))
            pen += np.sum(np.maximum(x + r_init - w, 0))
            pen += np.sum(np.maximum(r_init - y, 0))
            pen += np.sum(np.maximum(y + r_init - h, 0))
            dx = x[:, None] - x[None, :]
            dy = y[:, None] - y[None, :]
            d_sq = dx * dx + dy * dy
            r_sum = r_init[:, None] + r_init[None, :]
            iu = np.triu_indices(N, k=1)
            pen += np.sum(np.maximum(r_sum[iu] - np.sqrt(np.maximum(d_sq[iu], 0)), 0))
            return pen
        # Valid: try to maximize scale
        s = _max_scale_vec(x, y, w, h, r_norm)
        return -s

    # Initial guess: grid
    side = int(np.ceil(np.sqrt(N)))
    x0 = np.array([(c + 0.5) / side * w for c in range(N)])
    y0 = np.array([(r + 0.5) / side * h for r in range(N)])
    x0 = np.clip(x0, r_init, w - r_init)
    y0 = np.clip(y0, r_init, h - r_init)

    try:
        result = differential_evolution(
            pos_objective, bounds,
            seed=42 + int(w * 100),
            maxiter=80,
            popsize=12,
            tol=1e-8,
            polish=False,
            init='latinhypercube'
        )
        x_best = result.x[:N]
        y_best = result.x[N:2 * N]

        # Get actual radii at best scale
        s_best = _max_scale_vec(x_best, y_best, w, h, r_norm)
        r_best = r_norm * s_best

        if not _is_valid_packing(x_best, y_best, r_best, w, h):
            # Try with slightly smaller
            for shrink in [0.99, 0.98, 0.97, 0.95]:
                r_try = r_best * shrink
                if _is_valid_packing(x_best, y_best, r_try, w, h):
                    r_best = r_try
                    break
            else:
                return None

        # Polish
        polished = _polish_positions(x_best, y_best, r_best, w, h, time_budget)
        if polished is not None:
            px, py, pr, pw, ph = polished
            return px, py, pr, pw, ph
        else:
            return x_best, y_best, r_best, w, h

    except Exception:
        return None


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin

    best_value = 0.0
    best_result = None

    # Try multiple aspect ratios
    aspect_ratios = [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]

    # Base radius distribution: slightly varied for diversity
    rng = np.random.default_rng(123)
    r_base = 1.0 + rng.uniform(-0.1, 0.1, N)

    for w in aspect_ratios:
        if time.time() > deadline - 30:
            break

        time_budget = deadline - 5
        result = _try_aspect_ratio(w, None, None, r_base, time_budget)

        if result is not None:
            x, y, r, ww, hh = result
            if _is_valid_packing(x, y, r, ww, hh):
                val = float(np.sum(r))
                if val > best_value:
                    best_value = val
                    best_result = (x.copy(), y.copy(), r.copy(), ww, hh)

    # If we have a result, do a final polish pass
    if best_result is not None and time.time() < deadline - 10:
        x, y, r, w, h = best_result
        polished = _polish_positions(x, y, r, w, h, deadline - 2)
        if polished is not None:
            px, py, pr, pw, ph = polished
            if _is_valid_packing(px, py, pr, pw, ph):
                val = float(np.sum(pr))
                if val > best_value:
                    best_value = val
                    best_result = (px, py, pr, pw, ph)

    # Fallback
    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        w = 1.0
        h = 1.0
        r0 = 0.5 / side - 0.001
        x = np.array([(c + 0.5) / side * w for c in range(N)])
        y = np.array([(rr + 0.5) / side * h for rr in range(N)])
        r = np.full(N, r0)
        return np.column_stack((x, y, r))

    x, y, r, w, h = best_result

    # Final safety: tiny shrink
    for _ in range(5):
        r_try = r * (1 - 1e-9)
        if _is_valid_packing(x, y, r_try, w, h):
            r = r_try
        else:
            break

    return np.column_stack((x, y, r))
