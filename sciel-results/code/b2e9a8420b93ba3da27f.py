import numpy as np
from scipy.optimize import minimize, linprog
import time

N = 32
TIME_LIMIT = 138.0  # Evaluator kills at 150s; keep 12s margin

def _elapsed(start):
    return time.time() - start

def _is_valid(x, y, r, eps=1e-7):
    if np.any(x < -eps) or np.any(x > 1.0 + eps):
        return False
    if np.any(y < -eps) or np.any(y > 1.0 + eps):
        return False
    if np.any(r < -eps):
        return False
    if np.any(x < r - eps) or np.any(x > 1.0 - r + eps):
        return False
    if np.any(y < r - eps) or np.any(y > 1.0 - r + eps):
        return False
    for i in range(N):
        for j in range(i + 1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d = np.hypot(dx, dy)
            if d < (r[i] + r[j]) - eps:
                return False
    return True

def _sum_r(r):
    return float(np.sum(r))

def _optimize_radii_lp(x, y):
    """
    Solve for maximum radii given fixed centers using Linear Programming.
    Maximize sum(r) subject to r_i + r_j <= d_ij and r_i <= dist_to_bound.
    """
    c = -np.ones(N)
    A_ub = []
    b_ub = []
    
    # Pairwise constraints
    for i in range(N):
        for j in range(i+1, N):
            d = np.hypot(x[i]-x[j], y[i]-y[j])
            row = np.zeros(N)
            row[i] = 1.0
            row[j] = 1.0
            A_ub.append(row)
            b_ub.append(d)
            
    # Boundary constraints
    for i in range(N):
        b = min(x[i], 1.0-x[i], y[i], 1.0-y[i])
        row = np.zeros(N)
        row[i] = 1.0
        A_ub.append(row)
        b_ub.append(b)
        
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    bounds = [(0, None)] * N
    
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            return res.x
    except Exception:
        pass
        
    # Fallback: iterative shrink
    r = np.array([min(x[i], 1.0-x[i], y[i], 1.0-y[i]) for i in range(N)])
    for _ in range(20):
        changed = False
        for i in range(N):
            for j in range(i+1, N):
                d = np.hypot(x[i]-x[j], y[i]-y[j])
                if r[i] + r[j] > d:
                    shrink = (d / (r[i] + r[j])) * 0.999 if (r[i] + r[j]) > 0 else 0.5
                    r[i] *= shrink
                    r[j] *= shrink
                    changed = True
        if not changed: 
            break
    return r

def _make_valid(x, y, r, max_iter=200):
    """Iteratively shrink radii to satisfy all constraints."""
    x = x.copy()
    y = y.copy()
    r = r.copy()
    for _ in range(max_iter):
        changed = False
        for i in range(N):
            # Bound constraints
            r[i] = min(r[i], x[i] * 0.999, (1.0 - x[i]) * 0.999,
                       y[i] * 0.999, (1.0 - y[i]) * 0.999)
            if r[i] < 1e-6:
                r[i] = 1e-6
        # Pairwise overlap
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.hypot(dx, dy)
                min_d = r[i] + r[j]
                if d < min_d - 1e-10:
                    # Shrink both proportionally
                    shrink = (d / min_d) * 0.999 if min_d > 0 else 0.5
                    r[i] *= shrink
                    r[j] *= shrink
                    changed = True
        if not changed:
            break
    return x, y, r

def _hex_init():
    """Hexagonal packing initialization for 32 circles."""
    rows = [6, 5, 6, 5, 6, 4]  # 32 total
    x = []
    y = []
    r = []
    total_rows = len(rows)
    row_h = 1.0 / total_rows
    for row_idx, count in enumerate(rows):
        ry = (row_idx + 0.5) * row_h
        offset = 0.0
        if row_idx % 2 == 1:
            offset = 0.5 / count
        for c in range(count):
            cx = (c + 0.5 + offset) / count
            cr = 0.48 * row_h
            cr = min(cr, cx * 0.99, (1.0 - cx) * 0.99, ry * 0.99, (1.0 - ry) * 0.99)
            x.append(cx)
            y.append(ry)
            r.append(cr)
    x = x[:N]
    y = y[:N]
    r = r[:N]
    while len(x) < N:
        x.append(0.5)
        y.append(0.5)
        r.append(0.02)
    return np.array(x[:N]), np.array(y[:N]), np.array(r[:N])

def _rect_init():
    """Rectangular grid initialization."""
    side = int(np.ceil(np.sqrt(N)))
    x = []
    y = []
    r = []
    for i in range(side):
        for j in range(side):
            if len(x) < N:
                cx = (i + 0.5) / side
                cy = (j + 0.5) / side
                cr = 0.48 / side
                cr = min(cr, cx * 0.99, (1.0 - cx) * 0.99, cy * 0.99, (1.0 - cy) * 0.99)
                x.append(cx)
                y.append(cy)
                r.append(cr)
    return np.array(x[:N]), np.array(y[:N]), np.array(r[:N])

def _random_init(seed):
    """Random initialization."""
    rng = np.random.RandomState(seed)
    x = rng.uniform(0.1, 0.9, N)
    y = rng.uniform(0.1, 0.9, N)
    r = np.full(N, 0.06)
    return x, y, r

def _objective(params, start, soft_eps=1e-4, mu=50.0):
    if _elapsed(start) > TIME_LIMIT:
        return 1e10
    x = params[:N]
    y = params[N:2*N]
    r = params[2*N:3*N]
    r = np.maximum(r, 0.0)
    obj = -np.sum(r)

    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist = np.sqrt(dx*dx + dy*dy + 1e-14)
    min_dist = r[:, None] + r[None, :]
    diff = min_dist - dist
    smooth_max = 0.5 * (diff + np.sqrt(diff*diff + soft_eps*soft_eps))
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    overlap_pen = np.sum(smooth_max[mask])

    dneg_x = r - x
    dpos_x = x - (1.0 - r)
    dneg_y = r - y
    dpos_y = y - (1.0 - r)
    def sm(v):
        return 0.5 * (v + np.sqrt(v*v + soft_eps*soft_eps))
    bound_pen = np.sum(sm(dneg_x)) + np.sum(sm(dpos_x)) + np.sum(sm(dneg_y)) + np.sum(sm(dpos_y))

    return obj + mu * (overlap_pen + bound_pen)

def _objective_tight(params, start, soft_eps=1e-6, mu=200.0):
    if _elapsed(start) > TIME_LIMIT:
        return 1e10
    x = params[:N]
    y = params[N:2*N]
    r = params[2*N:3*N]
    r = np.maximum(r, 0.0)
    obj = -np.sum(r)

    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist = np.sqrt(dx*dx + dy*dy + 1e-14)
    min_dist = r[:, None] + r[None, :]
    diff = min_dist - dist
    smooth_max = 0.5 * (diff + np.sqrt(diff*diff + soft_eps*soft_eps))
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    overlap_pen = np.sum(smooth_max[mask])

    dneg_x = r - x
    dpos_x = x - (1.0 - r)
    dneg_y = r - y
    dpos_y = y - (1.0 - r)
    def sm(v):
        return 0.5 * (v + np.sqrt(v*v + soft_eps*soft_eps))
    bound_pen = np.sum(sm(dneg_x)) + np.sum(sm(dpos_x)) + np.sum(sm(dneg_y)) + np.sum(sm(dpos_y))

    return obj + mu * (overlap_pen + bound_pen)

def _push_apart(x, y, r, iterations=50):
    """Physics-based push apart to resolve overlaps."""
    x = x.copy()
    y = y.copy()
    for _ in range(iterations):
        moved = False
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.hypot(dx, dy)
                min_d = r[i] + r[j]
                if d < min_d - 1e-10:
                    if d < 1e-10:
                        dx, dy = 1e-3, 0.0
                        d = np.hypot(dx, dy)
                    # Push apart
                    overlap = min_d - d
                    nx, ny = dx/d, dy/d
                    push = overlap * 0.5 * 0.8
                    x[i] += nx * push
                    y[i] += ny * push
                    x[j] -= nx * push
                    y[j] -= ny * push
                    moved = True
        # Clamp to bounds
        for i in range(N):
            x[i] = np.clip(x[i], r[i]*0.99, 1.0 - r[i]*0.99)
            y[i] = np.clip(y[i], r[i]*0.99, 1.0 - r[i]*0.99)
        if not moved:
            break
    return x, y

def _optimize_with_lp_fix(x, y, r, start, max_iter=300):
    """
    Iteratively optimize positions using penalty methods, then fix radii via LP.
    This loop allows positions to adapt to the 'true' maximum radii.
    """
    best_x, best_y, best_r = x.copy(), y.copy(), r.copy()
    best_score = _sum_r(r)
    
    # Initial LP fix to get optimal radii for starting positions
    r = _optimize_radii_lp(x, y)
    if _is_valid(x, y, r):
        s = _sum_r(r)
        if s > best_score:
            best_score = s
            best_x, best_y, best_r = x.copy(), y.copy(), r.copy()
    
    for it in range(max_iter):
        if _elapsed(start) > TIME_LIMIT - 10:
            break
            
        # 1. Optimize positions (and radii) using penalty method
        # We use current radii as a starting point, but the optimizer will adjust them
        # to satisfy constraints. However, the penalty method might keep radii small
        # to avoid overlap penalties easily.
        p0 = np.concatenate([x, y, r])
        
        try:
            # Use L-BFGS-B for fast convergence
            bounds = []
            for i in range(N):
                bounds.append((0.0, 1.0)) # x
            for i in range(N):
                bounds.append((0.0, 1.0)) # y
            for i in range(N):
                bounds.append((0.0, 0.45)) # r
                
            res = minimize(
                _objective, p0, args=(start, 1e-4, 50.0),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-6, 'maxfun': 200}
            )
            x_new = res.x[:N]
            y_new = res.x[N:2*N]
            r_new = res.x[2*N:3*N]
            
            # 2. Fix radii via LP for these new positions
            r_lp = _optimize_radii_lp(x_new, y_new)
            
            # Check validity
            if _is_valid(x_new, y_new, r_lp):
                s = _sum_r(r_lp)
                if s > best_score:
                    best_score = s
                    best_x, best_y, best_r = x_new.copy(), y_new.copy(), r_lp.copy()
                    x, y, r = x_new, y_new, r_lp
                else:
                    # If LP didn't improve, try to move positions back towards best
                    # or just continue with current x,y but updated r from penalty
                    # Actually, if LP score is worse, it means positions are worse.
                    # We should probably revert x,y to best and try a different direction
                    # or just break if we are stagnating.
                    # For now, keep x,y as found by optimizer but use LP radii for next iter
                    # if they are valid, even if not best yet.
                    x, y, r = x_new, y_new, r_lp
            else:
                # Invalid, try to make valid
                x_v, y_v, r_v = _make_valid(x_new, y_new, r_lp)
                if _is_valid(x_v, y_v, r_v):
                    s = _sum_r(r_v)
                    if s > best_score:
                        best_score = s
                        best_x, best_y, best_r = x_v.copy(), y_v.copy(), r_v.copy()
                    x, y, r = x_v, y_v, r_v
                else:
                    # Revert to best
                    x, y, r = best_x, best_y, best_r
        except Exception:
            pass
            
    return best_x, best_y, best_r

def _tight_polish(x, y, r, start):
    best = (x, y, r)
    best_score = _sum_r(r)
    
    # 1. Tight L-BFGS-B
    if _elapsed(start) < TIME_LIMIT - 5:
        try:
            p0 = np.concatenate([best[0], best[1], best[2]])
            bounds = []
            for i in range(N):
                bounds.append((0.0, 1.0))
            for i in range(N):
                bounds.append((0.0, 1.0))
            for i in range(N):
                bounds.append((0.0, 0.45))
            res = minimize(
                _objective_tight, p0, args=(start, 1e-6, 200.0),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 5000, 'ftol': 1e-18, 'gtol': 1e-10, 'maxfun': 20000}
            )
            vx, vy, vr = _make_valid(res.x[:N], res.x[N:2*N], res.x[2*N:3*N])
            if _is_valid(vx, vy, vr):
                s = _sum_r(vr)
                if s > best_score:
                    best = (vx, vy, vr)
                    best_score = s
        except Exception:
            pass
    
    # 2. LP Fix on best
    if _elapsed(start) < TIME_LIMIT - 3:
        try:
            r_lp = _optimize_radii_lp(best[0], best[1])
            if _is_valid(best[0], best[1], r_lp):
                s = _sum_r(r_lp)
                if s > best_score:
                    best = (best[0], best[1], r_lp)
                    best_score = s
        except Exception:
            pass
    
    # 3. Powell with tight objective
    if _elapsed(start) < TIME_LIMIT - 3:
        try:
            p0 = np.concatenate([best[0], best[1], best[2]])
            res = minimize(
                _objective_tight, p0, args=(start, 1e-6, 200.0),
                method='Powell',
                options={'maxiter': 3000, 'ftol': 1e-15}
            )
            vx, vy, vr = _make_valid(res.x[:N], res.x[N:2*N], res.x[2*N:3*N])
            if _is_valid(vx, vy, vr):
                s = _sum_r(vr)
                if s > best_score:
                    best = (vx, vy, vr)
                    best_score = s
        except Exception:
            pass
            
    # 4. LP Fix again
    if _elapsed(start) < TIME_LIMIT - 2:
        try:
            r_lp = _optimize_radii_lp(best[0], best[1])
            if _is_valid(best[0], best[1], r_lp):
                s = _sum_r(r_lp)
                if s > best_score:
                    best = (best[0], best[1], r_lp)
                    best_score = s
        except Exception:
            pass
            
    # 5. Local perturbation + LP
    if _elapsed(start) < TIME_LIMIT - 1:
        rng = np.random.RandomState(42)
        for trial in range(50):
            if _elapsed(start) > TIME_LIMIT - 1:
                break
            scale = 0.005 * (1.0 - trial / 50.0)
            px = best[0] + rng.normal(0, scale, N)
            py = best[1] + rng.normal(0, scale, N)
            
            # Solve LP for these new positions
            r_lp = _optimize_radii_lp(px, py)
            
            if _is_valid(px, py, r_lp):
                s = _sum_r(r_lp)
                if s > best_score:
                    best = (px, py, r_lp)
                    best_score = s
    
    return best

def construct():
    start = time.time()
    
    # Generate diverse initializations
    inits = []
    inits.append(_hex_init())
    inits.append(_rect_init())
    inits.append(_random_init(1))
    inits.append(_random_init(7))
    inits.append(_random_init(42))
    
    best = inits[0]
    best_score = _sum_r(best[2])
    if not _is_valid(*best):
        best = _make_valid(*best)
        best_score = _sum_r(best[2])
    
    # Optimize each initialization
    # We allocate time based on number of inits
    n_inits = len(inits)
    time_per_init = (TIME_LIMIT - 20) / n_inits
    
    for k, (x, y, r) in enumerate(inits):
        if _elapsed(start) > TIME_LIMIT - 20:
            break
            
        # Ensure valid start
        x, y, r = _make_valid(x, y, r)
        
        # Run the LP-optimized loop
        res = _optimize_with_lp_fix(x, y, r, start, max_iter=100)
        
        if _is_valid(*res):
            s = _sum_r(res[2])
            if s > best_score:
                best_score = s
                best = res
    
    # Final high-precision polish on the best result
    if _elapsed(start) < TIME_LIMIT - 10:
        res = _tight_polish(*best, start)
        if _is_valid(*res):
            s = _sum_r(res[2])
            if s > best_score:
                best = res
                best_score = s
    
    # Final LP check
    if _elapsed(start) < TIME_LIMIT - 2:
        try:
            r_lp = _optimize_radii_lp(best[0], best[1])
            if _is_valid(best[0], best[1], r_lp):
                s = _sum_r(r_lp)
                if s > best_score:
                    best = (best[0], best[1], r_lp)
        except Exception:
            pass
    
    x, y, r = best
    x, y, r = _make_valid(x, y, r)
    result = np.column_stack((x, y, r))
    return result
