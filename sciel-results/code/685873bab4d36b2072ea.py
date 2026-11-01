"""Circle packing in perimeter-4 rectangle: gradient-free Nelder-Mead + greedy expansion."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21


def _is_valid(cx, cy, r, W, H, tol=1e-9):
    """Check if packing is valid."""
    if W <= 0.05 or H <= 0.05:
        return False
    if np.any(cx - r < -tol) or np.any(cy - r < -tol):
        return False
    if np.any(cx + r > W + tol) or np.any(cy + r > H + tol):
        return False
    for i in range(N):
        for j in range(i + 1, N):
            dx = cx[i] - cx[j]
            dy = cy[i] - cy[j]
            if dx * dx + dy * dy < (r[i] + r[j]) ** 2 - tol:
                return False
    return True


def _penalized_objective(z, penalty_weight=5000.0):
    """Penalized objective for gradient-free optimization."""
    x = z[0::3][:N]
    y = z[1::3][:N]
    r = z[2::3][:N]
    w = z[-1]
    h = 2.0 - w

    if w < 0.3 or h < 0.3 or np.any(r < 0.001):
        return 1e8

    obj = -np.sum(r)

    # Pairwise overlap
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d_sq = dx * dx + dy * dy
    r_sum = r[:, None] + r[None, :]
    iu = np.triu_indices(N, k=1)
    d_pairs = np.sqrt(np.maximum(d_sq[iu], 0.0))
    overlap = np.maximum(r_sum[iu] - d_pairs, 0.0)
    obj += penalty_weight * np.sum(overlap * overlap)

    # Boundary
    lv = np.maximum(r - x, 0.0)
    rv = np.maximum(x + r - w, 0.0)
    bv = np.maximum(r - y, 0.0)
    tv = np.maximum(y + r - h, 0.0)
    obj += penalty_weight * (np.sum(lv*lv) + np.sum(rv*rv) + np.sum(bv*bv) + np.sum(tv*tv))

    # Aspect ratio penalty
    if w < 0.6:
        obj += penalty_weight * (0.6 - w) ** 2
    elif w > 1.4:
        obj += penalty_weight * (w - 1.4) ** 2

    return obj


def _greedy_expand(x, y, r, w, h, time_budget):
    """Iteratively expand each circle to its maximum possible radius."""
    x = x.copy().astype(float)
    y = y.copy().astype(float)
    r = r.copy().astype(float)
    improved = True
    rounds = 0
    while improved and rounds < 50:
        if time.time() > time_budget:
            break
        improved = False
        rounds += 1
        for i in range(N):
            # Max radius from walls
            r_wall = min(x[i], w - x[i], y[i], h - y[i])
            r_max = r_wall
            # Max radius from other circles
            for j in range(N):
                if i == j:
                    continue
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx*dx + dy*dy)
                r_allowed = d - r[j]
                if r_allowed < r_max:
                    r_max = r_allowed
            r_max = max(0.0, r_max)
            if r_max > r[i] + 1e-8:
                r[i] = r_max
                improved = True
    return x, y, r


def _local_search_positions(x, y, r, w, h, time_budget):
    """Try small random perturbations of positions to find better packings."""
    x = x.copy().astype(float)
    y = y.copy().astype(float)
    r = r.copy().astype(float)
    rng = np.random.default_rng(123)
    best_sum = np.sum(r)
    
    for _ in range(5000):
        if time.time() > time_budget:
            break
        # Pick a random circle and nudge it
        i = rng.integers(0, N)
        dx = rng.normal(0, 0.005)
        dy = rng.normal(0, 0.005)
        nx = x[i] + dx
        ny = y[i] + dy
        
        # Check validity of new position
        if nx < r[i] or nx > w - r[i] or ny < r[i] or ny > h - r[i]:
            continue
        valid = True
        for j in range(N):
            if i == j:
                continue
            ddx = nx - x[j]
            ddy = ny - y[j]
            if ddx*ddx + ddy*ddy < (r[i] + r[j])**2 - 1e-9:
                valid = False
                break
        if valid:
            x[i] = nx
            y[i] = ny
            # Try to expand all circles
            x2, y2, r2 = _greedy_expand(x, y, r, w, h, time_budget)
            if np.sum(r2) > best_sum:
                best_sum = np.sum(r2)
                r = r2
                x = x2
                y = y2
            else:
                # Revert
                x[i] -= dx
                y[i] -= dy
                r = r.copy()
    return x, y, r


def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    best_value = 0.0
    best_result = None

    def check_validity(z):
        x = z[0::3][:N]
        y = z[1::3][:N]
        r = z[2::3][:N]
        w = z[-1]
        h = 2.0 - w
        if _is_valid(x, y, r, w, h):
            return True, float(np.sum(r))
        return False, 0.0

    # Generate diverse initial guesses
    def get_initial_guesses():
        guesses = []
        side = int(np.ceil(np.sqrt(N)))
        
        # Grid-based initial guesses with various aspect ratios
        aspect_ratios = [1.0, 0.9, 1.1, 0.8, 1.25, 0.75, 1.33, 0.7, 1.5, 0.65]
        for w0 in aspect_ratios:
            h0 = 2.0 - w0
            if h0 < 0.35:
                continue
            z = np.zeros(3 * N + 1)
            for i in range(N):
                row, col = divmod(i, side)
                cx = (col + 0.5) / side * w0
                cy = (row + 0.5) / side * h0
                r0 = min(w0 / side, h0 / side) * 0.42
                z[3*i] = cx
                z[3*i+1] = cy
                z[3*i+2] = r0
            z[-1] = w0
            guesses.append(z)
        
        # Random starts
        rng = np.random.default_rng(42)
        for _ in range(15):
            w0 = rng.uniform(0.7, 1.3)
            h0 = 2.0 - w0
            z = np.zeros(3 * N + 1)
            for i in range(N):
                r0 = rng.uniform(0.04, 0.10)
                cx = rng.uniform(r0, w0 - r0)
                cy = rng.uniform(r0, h0 - r0)
                z[3*i] = cx
                z[3*i+1] = cy
                z[3*i+2] = r0
            z[-1] = w0
            guesses.append(z)
        
        return guesses

    guesses = get_initial_guesses()

    for i, z0 in enumerate(guesses):
        if time.time() > deadline - 15:
            break
        
        # Nelder-Mead optimization (gradient-free)
        remaining = deadline - (time.time() - start_time)
        maxiter = int(min(3000, max(200, remaining * 40)))
        
        try:
            res = minimize(_penalized_objective, z0, method='Nelder-Mead',
                           options={'maxiter': maxiter, 'xatol': 1e-9, 'fatol': 1e-9,
                                    'adaptive': True})
            z_opt = res.x
            valid, val = check_validity(z_opt)
            
            if valid and val > best_value:
                best_value = val
                best_result = z_opt.copy()
            elif not valid:
                # Try to repair by shrinking radii
                z_repair = z_opt.copy()
                r_idx = np.arange(2, len(z_repair)-1, 3)
                z_repair[r_idx] *= 0.9
                valid, val = check_validity(z_repair)
                if valid and val > best_value:
                    best_value = val
                    best_result = z_repair.copy()
        except Exception:
            continue

    if best_result is None:
        # Fallback
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i+0.5)/side, (j+0.5)/side, 0.5/side - 1e-3)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    # Polish: greedy expansion + local search
    if time.time() < deadline - 10:
        x = best_result[0::3][:N].copy()
        y = best_result[1::3][:N].copy()
        r = best_result[2::3][:N].copy()
        w = best_result[-1]
        h = 2.0 - w
        
        # Greedy expansion
        x, y, r = _greedy_expand(x, y, r, w, h, deadline - 5)
        if _is_valid(x, y, r, w, h):
            val = float(np.sum(r))
            if val > best_value:
                best_value = val
                best_result = np.empty(3*N+1)
                best_result[0::3][:N] = x
                best_result[1::3][:N] = y
                best_result[2::3][:N] = r
                best_result[-1] = w

        # Local search on positions
        if time.time() < deadline - 5:
            x = best_result[0::3][:N].copy()
            y = best_result[1::3][:N].copy()
            r = best_result[2::3][:N].copy()
            w = best_result[-1]
            h = 2.0 - w
            x, y, r = _local_search_positions(x, y, r, w, h, deadline - 3)
            if _is_valid(x, y, r, w, h):
                val = float(np.sum(r))
                if val > best_value:
                    best_value = val
                    best_result = np.empty(3*N+1)
                    best_result[0::3][:N] = x
                    best_result[1::3][:N] = y
                    best_result[2::3][:N] = r
                    best_result[-1] = w

        # Final greedy expansion
        if time.time() < deadline - 2:
            x = best_result[0::3][:N].copy()
            y = best_result[1::3][:N].copy()
            r = best_result[2::3][:N].copy()
            w = best_result[-1]
            h = 2.0 - w
            x, y, r = _greedy_expand(x, y, r, w, h, deadline - 1)
            if _is_valid(x, y, r, w, h):
                val = float(np.sum(r))
                if val > best_value:
                    best_value = val
                    best_result = np.empty(3*N+1)
                    best_result[0::3][:N] = x
                    best_result[1::3][:N] = y
                    best_result[2::3][:N] = r
                    best_result[-1] = w

    # Format output
    x = best_result[0::3][:N]
    y = best_result[1::3][:N]
    r = best_result[2::3][:N]
    w = best_result[-1]
    h = 2.0 - w

    # Final safety shrink
    for _ in range(5):
        r_trial = r * (1 - 1e-10)
        if _is_valid(x, y, r_trial, w, h):
            r = r_trial
        else:
            break

    return np.column_stack((x, y, r))
