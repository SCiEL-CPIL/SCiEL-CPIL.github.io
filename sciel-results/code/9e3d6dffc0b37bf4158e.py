import numpy as np
from scipy.optimize import minimize, linprog
import time

N = 32
TIME_LIMIT = 140.0  # Evaluator kills at 150s; keep 10s margin

def _elapsed(start):
    return time.time() - start

def _solve_radii_lp(x, y):
    """
    Solve for maximum radii given fixed centers using Linear Programming.
    Maximize sum(r) subject to:
    1. r_i + r_j <= d_ij (non-overlap)
    2. r_i <= dist_to_boundary_i (inside square)
    3. r_i >= 0
    """
    c = -np.ones(N)
    A_ub = []
    b_ub = []
    
    # Pairwise constraints: r_i + r_j <= d_ij
    # Vectorized distance calculation for speed
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist = np.sqrt(dx*dx + dy*dy)
    
    # Only need upper triangle for constraints
    for i in range(N):
        for j in range(i+1, N):
            row = np.zeros(N)
            row[i] = 1.0
            row[j] = 1.0
            A_ub.append(row)
            b_ub.append(dist[i, j])
            
    # Boundary constraints: r_i <= min(x_i, 1-x_i, y_i, 1-y_i)
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
        
    # Fallback: simple iterative shrink if LP fails (should rarely happen)
    r = np.array([min(x[i], 1.0-x[i], y[i], 1.0-y[i]) for i in range(N)])
    for _ in range(50):
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

def _is_valid(x, y, r, eps=1e-8):
    """Check if the packing is valid."""
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

def _objective_centers(params, start, p=10.0):
    """
    Objective function to maximize the minimum distance between points.
    We minimize - (sum of inverse powers of distances) to encourage spreading.
    This is a smooth surrogate for the max-min problem.
    """
    if _elapsed(start) > TIME_LIMIT:
        return 1e10
    
    x = params[:N]
    y = params[N:2*N]
    
    # Distances between pairs
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist = np.sqrt(dx*dx + dy*dy + 1e-12)
    
    # Distances to boundaries
    d_bound = np.minimum(np.minimum(x, 1.0-x), np.minimum(y, 1.0-y))
    
    # We want to maximize the minimum of {dist_ij, d_bound_i}
    # Approximation: Minimize sum(1/d^p) for all constraints
    # Mask out diagonal for pair distances
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    
    # Pairwise term
    pair_dist = dist[mask]
    pair_term = np.sum(1.0 / (pair_dist ** p))
    
    # Boundary term
    bound_term = np.sum(1.0 / (d_bound ** p))
    
    # Return negative to maximize
    return -(pair_term + bound_term)

def _optimize_centers(x0, start, method='L-BFGS-B', maxiter=5000):
    """
    Optimize center positions only.
    """
    try:
        res = minimize(lambda p: _objective_centers(p, start), x0, method=method,
                       options={'maxiter': maxiter, 'ftol': 1e-15, 'gtol': 1e-12,
                                'maxfun': maxiter * 2})
        return res.x
    except Exception:
        return x0

def _hex_init():
    """Hexagonal packing initialization."""
    # Approximate hex packing for 32 circles
    # 6-5-6-5-6-4 rows
    rows = [6, 5, 6, 5, 6, 4]
    x = []
    y = []
    total_rows = len(rows)
    row_h = 1.0 / total_rows
    for row_idx, count in enumerate(rows):
        ry = (row_idx + 0.5) * row_h
        offset = 0.0
        if row_idx % 2 == 1:
            offset = 0.5 / count
        for c in range(count):
            cx = (c + 0.5 + offset) / count
            x.append(cx)
            y.append(ry)
    x = x[:N]
    y = y[:N]
    while len(x) < N:
        x.append(0.5)
        y.append(0.5)
    return np.concatenate([x[:N], y[:N]])

def _grid_init(cols, rows):
    """Grid initialization."""
    x = []
    y = []
    for i in range(rows):
        for j in range(cols):
            if len(x) < N:
                x.append((j + 0.5) / cols)
                y.append((i + 0.5) / rows)
    while len(x) < N:
        x.append(0.5)
        y.append(0.5)
    return np.concatenate([x[:N], y[:N]])

def _random_init(seed):
    """Random initialization."""
    rng = np.random.RandomState(seed)
    x = rng.uniform(0.1, 0.9, N)
    y = rng.uniform(0.1, 0.9, N)
    return np.concatenate([x, y])

def construct():
    start_time = time.time()
    
    # 1. Generate diverse initializations for centers
    inits = []
    
    # Hexagonal
    inits.append(_hex_init())
    
    # Grids
    inits.append(_grid_init(6, 6)) # 36 slots, take 32
    inits.append(_grid_init(5, 7)) # 35 slots
    inits.append(_grid_init(4, 8)) # 32 slots exactly
    
    # Randoms
    for seed in [42, 123, 456, 789, 101, 202, 303, 404, 505, 606]:
        inits.append(_random_init(seed))
        
    best_score = -np.inf
    best_params = None
    
    # 2. Iterate through initializations
    for i, x0 in enumerate(inits):
        if _elapsed(start_time) > TIME_LIMIT - 20:
            break
            
        # Optimize centers
        opt_params = _optimize_centers(x0, start_time, maxiter=3000)
        
        # Extract centers
        x = opt_params[:N]
        y = opt_params[N:2*N]
        
        # Solve for optimal radii using LP
        r = _solve_radii_lp(x, y)
        
        # Check validity
        if not _is_valid(x, y, r):
            # If invalid, shrink radii slightly to ensure validity
            # This is a safety net; LP should usually be valid
            scale = 0.999
            r = r * scale
            if not _is_valid(x, y, r):
                # Fallback: force valid by clipping
                for k in range(N):
                    r[k] = min(r[k], x[k], 1-x[k], y[k], 1-y[k])
                # Check overlaps again
                for k in range(N):
                    for m in range(k+1, N):
                        d = np.hypot(x[k]-x[m], y[k]-y[m])
                        if r[k] + r[m] > d:
                            shrink = d / (r[k] + r[m]) * 0.999
                            r[k] *= shrink
                            r[m] *= shrink
        
        score = _sum_r(r)
        
        if score > best_score:
            best_score = score
            best_params = np.concatenate([x, y, r])
            
    # 3. Local Search / Refinement on the best result
    if best_params is not None and _elapsed(start_time) < TIME_LIMIT - 10:
        rng = np.random.RandomState(999)
        current_best = best_params.copy()
        current_score = best_score
        
        # Perform a few rounds of jittering centers and re-optimizing
        for _ in range(20):
            if _elapsed(start_time) > TIME_LIMIT - 5:
                break
                
            # Jitter centers
            x = current_best[:N]
            y = current_best[N:2*N]
            jitter_scale = 0.02
            x_jit = x + rng.uniform(-jitter_scale, jitter_scale, N)
            y_jit = y + rng.uniform(-jitter_scale, jitter_scale, N)
            
            # Clip to [0,1]
            x_jit = np.clip(x_jit, 0, 1)
            y_jit = np.clip(y_jit, 0, 1)
            
            p_jit = np.concatenate([x_jit, y_jit])
            
            # Short optimization from jittered point
            opt_jit = _optimize_centers(p_jit, start_time, maxiter=500)
            
            x_j = opt_jit[:N]
            y_j = opt_jit[N:2*N]
            
            r_j = _solve_radii_lp(x_j, y_j)
            
            if not _is_valid(x_j, y_j, r_j):
                r_j = r_j * 0.999
                if not _is_valid(x_j, y_j, r_j):
                    continue
                    
            score_j = _sum_r(r_j)
            
            if score_j > current_score:
                current_score = score_j
                current_best = np.concatenate([x_j, y_j, r_j])
                
        if current_score > best_score:
            best_score = current_score
            best_params = current_best
            
    # 4. Final Polish
    if best_params is not None and _elapsed(start_time) < TIME_LIMIT - 5:
        # One more optimization pass on the best centers
        x_final = best_params[:N]
        y_final = best_params[N:2*N]
        p_final = np.concatenate([x_final, y_final])
        
        opt_final = _optimize_centers(p_final, start_time, maxiter=2000)
        x_f = opt_final[:N]
        y_f = opt_final[N:2*N]
        
        r_f = _solve_radii_lp(x_f, y_f)
        
        if _is_valid(x_f, y_f, r_f):
            score_f = _sum_r(r_f)
            if score_f > best_score:
                best_score = score_f
                best_params = np.concatenate([x_f, y_f, r_f])
        else:
            # Stick with previous valid result
            pass

    if best_params is None:
        # Emergency fallback
        x, y = _grid_init(4, 8)[:N], _grid_init(4, 8)[N:2*N]
        r = _solve_radii_lp(x, y)
        best_params = np.concatenate([x, y, r])
        
    x = best_params[:N]
    y = best_params[N:2*N]
    r = best_params[2*N:3*N]
    
    result = np.column_stack((x, y, r))
    return result
