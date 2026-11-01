import numpy as np
from scipy.optimize import minimize
import time

N = 32
TIME_LIMIT = 145.0  # Evaluator kills at 150s, keep 5s margin for safety

def _elapsed(start):
    return time.time() - start

def _is_valid(x, y, r, eps=1e-7):
    # bounds
    if np.any(x < -eps) or np.any(x > 1.0 + eps):
        return False
    if np.any(y < -eps) or np.any(y > 1.0 + eps):
        return False
    if np.any(r < -eps):
        return False
    # circle in square
    if np.any(x < r - eps) or np.any(x > 1.0 - r + eps):
        return False
    if np.any(y < r - eps) or np.any(y > 1.0 - r + eps):
        return False
    # pairwise disjoint interiors
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

def _objective(params, mu=50.0, eps=1e-4):
    """
    Vectorized objective function with soft penalties.
    Minimizes -sum(r) + mu * (overlap_penalty + boundary_penalty)
    """
    x = params[:N]
    y = params[N:2*N]
    r = params[2*N:3*N]
    
    # Objective: maximize sum of radii
    obj = -np.sum(r)
    
    # Pairwise overlap penalty
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist = np.sqrt(dx*dx + dy*dy + 1e-12)
    min_dist = r[:, None] + r[None, :]
    diff = min_dist - dist
    # Smooth max approximation: 0.5 * (diff + sqrt(diff^2 + eps^2))
    smooth_max = 0.5 * (diff + np.sqrt(diff*diff + eps*eps))
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    overlap_pen = np.sum(smooth_max[mask])
    
    # Boundary penalties
    # r - x < 0 means x < r (left violation)
    # x - (1-r) < 0 means x > 1-r (right violation)
    dneg_x = r - x
    dpos_x = x - (1.0 - r)
    dneg_y = r - y
    dpos_y = y - (1.0 - r)
    
    bound_pen = np.sum(0.5 * (dneg_x + np.sqrt(dneg_x*dneg_x + eps*eps))) + \
                np.sum(0.5 * (dpos_x + np.sqrt(dpos_x*dpos_x + eps*eps))) + \
                np.sum(0.5 * (dneg_y + np.sqrt(dneg_y*dneg_y + eps*eps))) + \
                np.sum(0.5 * (dpos_y + np.sqrt(dpos_y*dpos_y + eps*eps)))
    
    return obj + mu * (overlap_pen + bound_pen)

def _make_valid(x, y, r, max_iter=100):
    """
    Enforce validity by shrinking radii if constraints are violated.
    This is a fallback to ensure the returned solution is always valid.
    """
    x = x.copy()
    y = y.copy()
    r = r.copy()
    
    # First, clamp positions to be within bounds given current radii
    # This might cause overlaps, but ensures boundary validity
    x = np.clip(x, r, 1.0 - r)
    y = np.clip(y, r, 1.0 - r)
    
    for _ in range(max_iter):
        valid = True
        # Check boundaries
        for i in range(N):
            if x[i] < r[i] - 1e-9 or x[i] > 1.0 - r[i] + 1e-9 or \
               y[i] < r[i] - 1e-9 or y[i] > 1.0 - r[i] + 1e-9:
                # Shrink radius to fit boundary
                max_r = min(x[i], 1.0 - x[i], y[i], 1.0 - y[i])
                if r[i] > max_r:
                    r[i] = max(0.0, max_r * 0.999)
                    valid = False
        
        # Check overlaps
        for i in range(N):
            for j in range(i + 1, N):
                d = np.hypot(x[i] - x[j], y[i] - y[j])
                need = r[i] + r[j]
                if d < need - 1e-9:
                    # Shrink both proportionally to fit distance
                    # We want r_i + r_j = d. Let's shrink both by factor f
                    # f * r_i + f * r_j = d => f = d / (r_i + r_j)
                    f = (d / need) * 0.999 if need > 0 else 0.5
                    r[i] *= f
                    r[j] *= f
                    valid = False
        if valid:
            break
            
    return x, y, r

def _hex_init():
    """Hexagonal packing initialization."""
    # 6x6 grid is 36, we take 32.
    # Or better, a specific hex layout for 32.
    # Let's use a grid that approximates hex packing.
    x = np.zeros(N)
    y = np.zeros(N)
    r = np.zeros(N)
    
    # 6 rows, varying counts: 6, 5, 6, 5, 6, 4 = 32
    rows = [6, 5, 6, 5, 6, 4]
    idx = 0
    row_h = 1.0 / 6.0
    for row_idx, count in enumerate(rows):
        ry = (row_idx + 0.5) * row_h
        # Offset odd rows
        offset = 0.0
        if row_idx % 2 == 1:
            offset = 0.5 / count
        for c in range(count):
            if idx >= N:
                break
            cx = (c + 0.5 + offset) / count
            x[idx] = cx
            y[idx] = ry
            # Initial radius: half the min distance to neighbors or boundary
            # Approximate: half row height
            r[idx] = 0.48 * row_h
            idx += 1
            
    # Ensure within bounds
    for i in range(N):
        r[i] = min(r[i], x[i], 1.0-x[i], y[i], 1.0-y[i])
        
    return x, y, r

def _rect_init(cols, rows):
    """Rectangular grid initialization."""
    x = np.zeros(N)
    y = np.zeros(N)
    r = np.zeros(N)
    
    total = cols * rows
    # If total < N, we can't fill all, but we assume cols*rows >= N
    for j in range(rows):
        for i in range(cols):
            idx = j * cols + i
            if idx >= N:
                break
            x[idx] = (i + 0.5) / cols
            y[idx] = (j + 0.5) / rows
            r[idx] = 0.48 / max(cols, rows)
            
    # Ensure within bounds
    for i in range(N):
        r[i] = min(r[i], x[i], 1.0-x[i], y[i], 1.0-y[i])
        
    return x, y, r

def _random_init(seed):
    """Random initialization."""
    rng = np.random.RandomState(seed)
    x = rng.uniform(0.1, 0.9, N)
    y = rng.uniform(0.1, 0.9, N)
    r = np.full(N, 0.06)
    # Ensure within bounds
    for i in range(N):
        r[i] = min(r[i], x[i], 1.0-x[i], y[i], 1.0-y[i])
    return x, y, r

def _optimize_single(x, y, r, start, time_budget=20.0):
    """
    Run L-BFGS-B optimization on a single configuration.
    """
    params = np.concatenate([x, y, r])
    
    # Phase 1: Coarse optimization
    res = minimize(
        _objective, params, 
        method='L-BFGS-B',
        args=(50.0, 1e-4), # mu, eps
        options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 20000}
    )
    
    best_params = res.x
    best_score = _sum_r(best_params[2*N:3*N])
    
    # Check time
    if _elapsed(start) > TIME_LIMIT - time_budget:
        return best_params
    
    # Phase 2: Tight optimization
    # Use tighter mu and smaller eps to push constraints
    res2 = minimize(
        _objective, best_params, 
        method='L-BFGS-B',
        args=(200.0, 1e-6), # tighter
        options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10, 'maxfun': 10000}
    )
    
    if _sum_r(res2.x[2*N:3*N]) > best_score:
        best_params = res2.x
        best_score = _sum_r(best_params[2*N:3*N])
        
    return best_params

def construct():
    start = time.time()
    
    # Generate diverse initializations
    inits = []
    inits.append(_hex_init())
    inits.append(_rect_init(6, 6))
    inits.append(_rect_init(8, 5))
    inits.append(_rect_init(5, 7))
    inits.append(_rect_init(7, 5))
    inits.append(_random_init(1))
    inits.append(_random_init(42))
    inits.append(_random_init(123))
    
    best_params = None
    best_score = -np.inf
    
    # Optimize each initialization
    for x, y, r in inits:
        if _elapsed(start) > TIME_LIMIT - 10:
            break
            
        # Ensure initial validity to help optimizer
        x_v, y_v, r_v = _make_valid(x, y, r)
        
        params_opt = _optimize_single(x_v, y_v, r_v, start)
        
        # Finalize
        x_f = params_opt[:N]
        y_f = params_opt[N:2*N]
        r_f = params_opt[2*N:3*N]
        
        # Make valid
        x_f, y_f, r_f = _make_valid(x_f, y_f, r_f)
        
        if _is_valid(x_f, y_f, r_f):
            score = _sum_r(r_f)
            if score > best_score:
                best_score = score
                best_params = np.concatenate([x_f, y_f, r_f])
    
    # If no valid solution found (unlikely), return a simple valid one
    if best_params is None:
        x, y, r = _rect_init(6, 6)
        x, y, r = _make_valid(x, y, r)
        best_params = np.concatenate([x, y, r])
        best_score = _sum_r(r)
        
    # Final polish on the best solution if time permits
    if _elapsed(start) < TIME_LIMIT - 5:
        x_b = best_params[:N]
        y_b = best_params[N:2*N]
        r_b = best_params[2*N:3*N]
        
        # Very tight optimization
        res_final = minimize(
            _objective, best_params, 
            method='L-BFGS-B',
            args=(500.0, 1e-8), # very tight
            options={'maxiter': 200, 'ftol': 1e-15, 'gtol': 1e-12, 'maxfun': 5000}
        )
        
        x_f = res_final.x[:N]
        y_f = res_final.x[N:2*N]
        r_f = res_final.x[2*N:3*N]
        
        x_f, y_f, r_f = _make_valid(x_f, y_f, r_f)
        
        if _is_valid(x_f, y_f, r_f):
            score = _sum_r(r_f)
            if score > best_score:
                best_params = np.concatenate([x_f, y_f, r_f])
                best_score = score
                
    x, y, r = best_params[:N], best_params[N:2*N], best_params[2*N:3*N]
    
    # Final safety check
    x, y, r = _make_valid(x, y, r)
    
    result = np.column_stack((x, y, r))
    return result
