import numpy as np
from scipy.optimize import minimize
import time

N = 32
TIME_LIMIT = 138.0  # safety margin before 150s kill

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

def _make_valid(x, y, r, max_iter=300):
    """Iteratively shrink radii to satisfy all constraints."""
    x = x.copy()
    y = y.copy()
    r = r.copy()
    for _ in range(max_iter):
        changed = False
        for i in range(N):
            r[i] = min(r[i], x[i] * 0.999, (1.0 - x[i]) * 0.999,
                       y[i] * 0.999, (1.0 - y[i]) * 0.999)
            if r[i] < 1e-8:
                r[i] = 1e-8
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.hypot(dx, dy)
                min_d = r[i] + r[j]
                if d < min_d - 1e-10:
                    shrink = (d / min_d) * 0.999 if min_d > 0 else 0.5
                    r[i] *= shrink
                    r[j] *= shrink
                    changed = True
        if not changed:
            break
    return x, y, r

def _hex_init():
    """Hexagonal packing with 6-5-6-5-6-4 rows."""
    rows = [6, 5, 6, 5, 6, 4]
    x, y, r = [], [], []
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
    """Rectangular 6x6 grid."""
    x, y, r = [], [], []
    for i in range(N):
        row, col = i // 6, i % 6
        cx = (col + 0.5) / 6
        cy = (row + 0.5) / 6
        cr = 0.48 / 6
        cr = min(cr, cx * 0.99, (1.0 - cx) * 0.99, cy * 0.99, (1.0 - cy) * 0.99)
        x.append(cx)
        y.append(cy)
        r.append(cr)
    return np.array(x), np.array(y), np.array(r)

def _random_init(seed):
    rng = np.random.RandomState(seed)
    x = rng.uniform(0.1, 0.9, N)
    y = rng.uniform(0.1, 0.9, N)
    r = np.full(N, 0.06)
    return x, y, r

def _objective(params, soft_eps=1e-4):
    x = params[:N]
    y = params[N:2*N]
    r = params[2*N:3*N]
    
    obj = -np.sum(r)
    
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist = np.sqrt(dx*dx + dy*dy)
    min_dist = r[:, None] + r[None, :]
    diff = min_dist - dist
    smooth_max = 0.5 * (diff + np.sqrt(diff*diff + soft_eps*soft_eps))
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    overlap_pen = np.sum(smooth_max[mask])
    
    pen = np.zeros(4*N)
    pen[0:N] = r - x
    pen[N:2*N] = x - (1.0 - r)
    pen[2*N:3*N] = r - y
    pen[3*N:4*N] = y - (1.0 - r)
    bound_pen = np.sum(0.5 * (pen + np.sqrt(pen*pen + soft_eps*soft_eps)))
    
    mu = 50.0
    return obj + mu * (overlap_pen + bound_pen)

def _objective_tight(params, soft_eps=1e-6):
    x = params[:N]
    y = params[N:2*N]
    r = params[2*N:3*N]
    
    obj = -np.sum(r)
    
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist = np.sqrt(dx*dx + dy*dy)
    min_dist = r[:, None] + r[None, :]
    diff = min_dist - dist
    smooth_max = 0.5 * (diff + np.sqrt(diff*diff + soft_eps*soft_eps))
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    overlap_pen = np.sum(smooth_max[mask])
    
    pen = np.zeros(4*N)
    pen[0:N] = r - x
    pen[N:2*N] = x - (1.0 - r)
    pen[2*N:3*N] = r - y
    pen[3*N:4*N] = y - (1.0 - r)
    bound_pen = np.sum(0.5 * (pen + np.sqrt(pen*pen + soft_eps*soft_eps)))
    
    mu = 200.0
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
                    overlap = min_d - d
                    nx, ny = dx/d, dy/d
                    push = overlap * 0.5 * 0.8
                    x[i] += nx * push
                    y[i] += ny * push
                    x[j] -= nx * push
                    y[j] -= ny * push
                    moved = True
        for i in range(N):
            x[i] = np.clip(x[i], r[i]*0.99, 1.0 - r[i]*0.99)
            y[i] = np.clip(y[i], r[i]*0.99, 1.0 - r[i]*0.99)
        if not moved:
            break
    return x, y

def _optimize(x0, start_time, time_budget=100):
    try:
        res = minimize(_objective, x0, method='L-BFGS-B',
                       options={'maxiter': 3000, 'ftol': 1e-14, 'gtol': 1e-12,
                                'maxfun': 50000})
        return res.x
    except Exception:
        return x0

def _tight_polish(params, start_time, time_budget=10):
    try:
        res = minimize(_objective_tight, params, method='L-BFGS-B',
                       options={'maxiter': 5000, 'ftol': 1e-15, 'gtol': 1e-14,
                                'maxfun': 100000})
        return res.x
    except Exception:
        return params

def _local_search(params, start_time, max_iter=1000):
    """Local search with random perturbations to escape local minima."""
    best_params = params.copy()
    best_score = -float(np.sum(best_params[2*N:3*N]))
    rng = np.random.RandomState(12345)
    
    for i in range(max_iter):
        if _elapsed(start_time) > TIME_LIMIT - 10:
            break
        
        scale = 0.05 * (1 - i / max_iter)
        perturbed = best_params + rng.normal(0, scale, size=3*N)
        
        px = perturbed[:N].copy()
        py = perturbed[N:2*N].copy()
        pr = perturbed[2*N:3*N].copy()
        px, py = _push_apart(px, py, pr, iterations=20)
        p_opt = np.concatenate([px, py, pr])
        
        try:
            res = minimize(_objective, p_opt, method='L-BFGS-B',
                           options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10,
                                    'maxfun': 5000})
            p_new = res.x
        except Exception:
            continue
        
        px_new = p_new[:N]
        py_new = p_new[N:2*N]
        pr_new = p_new[2*N:3*N]
        
        if _is_valid(px_new, py_new, pr_new):
            score = -float(np.sum(pr_new))
            if score > best_score:
                best_score = score
                best_params = p_new.copy()
    
    return best_params

def _grow_radii(params, start_time, n_grow=20):
    """Iteratively grow radii and re-optimize."""
    best_params = params.copy()
    best_score = -float(np.sum(best_params[2*N:3*N]))
    
    for _ in range(n_grow):
        if _elapsed(start_time) > TIME_LIMIT - 15:
            break
        px = best_params[:N].copy()
        py = best_params[N:2*N].copy()
        pr = best_params[2*N:3*N].copy()
        
        pr_new = pr * 1.005
        p_trial = np.concatenate([px, py, pr_new])
        
        try:
            res = minimize(_objective, p_trial, method='L-BFGS-B',
                           options={'maxiter': 300, 'ftol': 1e-13, 'gtol': 1e-11,
                                    'maxfun': 3000})
            p_new = res.x
        except Exception:
            break
        
        px_new = p_new[:N]
        py_new = p_new[N:2*N]
        pr_new2 = p_new[2*N:3*N]
        
        if _is_valid(px_new, py_new, pr_new2):
            score = -float(np.sum(pr_new2))
            if score > best_score:
                best_score = score
                best_params = p_new.copy()
        else:
            break
    
    return best_params

def construct():
    start_time = time.time()
    
    best_params = None
    best_score = -np.inf
    
    # Phase 1: Multiple diverse restarts
    inits = [
        _hex_init,
        _rect_init,
        lambda: _random_init(42),
        lambda: _random_init(123),
        lambda: _random_init(999),
    ]
    
    for init_fn in inits:
        if _elapsed(start_time) > TIME_LIMIT - 30:
            break
        
        x, y, r = init_fn()
        x, y = _push_apart(x, y, r, iterations=30)
        params = np.concatenate([x, y, r])
        
        # Optimize
        optimized = _optimize(params, start_time)
        
        # Check validity
        ox = optimized[:N]
        oy = optimized[N:2*N]
        or_ = optimized[2*N:3*N]
        
        if _is_valid(ox, oy, or_):
            score = -float(np.sum(or_))
            if score > best_score:
                best_score = score
                best_params = optimized.copy()
        else:
            # Try to repair
            ox2, oy2, or2 = _make_valid(ox, oy, or_)
            if _is_valid(ox2, oy2, or2):
                score = -float(np.sum(or2))
                if score > best_score:
                    best_score = score
                    best_params = np.concatenate([ox2, oy2, or2])
    
    # Phase 2: Local search on best
    if best_params is not None and _elapsed(start_time) < TIME_LIMIT - 20:
        best_params = _local_search(best_params, start_time, max_iter=500)
        best_score = -float(np.sum(best_params[2*N:3*N]))
    
    # Phase 3: Grow radii
    if best_params is not None and _elapsed(start_time) < TIME_LIMIT - 15:
        best_params = _grow_radii(best_params, start_time, n_grow=15)
        best_score = -float(np.sum(best_params[2*N:3*N]))
    
    # Phase 4: Tight polish
    if best_params is not None and _elapsed(start_time) < TIME_LIMIT - 8:
        polished = _tight_polish(best_params, start_time)
        px = polished[:N]
        py = polished[N:2*N]
        pr = polished[2*N:3*N]
        if _is_valid(px, py, pr):
            score = -float(np.sum(pr))
            if score > best_score:
                best_score = score
                best_params = polished.copy()
    
    # Final: ensure validity and return
    if best_params is not None:
        x = best_params[:N].copy()
        y = best_params[N:2*N].copy()
        r = best_params[2*N:3*N].copy()
        
        # Final clip
        r = np.clip(r, 0, 0.5)
        x = np.clip(x, r, 1.0 - r)
        y = np.clip(y, r, 1.0 - r)
        
        if _is_valid(x, y, r):
            return np.column_stack([x, y, r])
        else:
            x, y, r = _make_valid(x, y, r)
            if _is_valid(x, y, r):
                return np.column_stack([x, y, r])
    
    # Fallback
    x, y, r = _rect_init()
    x, y, r = _make_valid(x, y, r)
    return np.column_stack([x, y, r])
