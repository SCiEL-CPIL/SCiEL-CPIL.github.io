import numpy as np
from scipy.optimize import minimize, linprog
import time

N = 32
TIME_LIMIT = 138.0  # evaluator kills at 150s; keep 12s margin

def _elapsed(start):
    return time.time() - start

def _optimize_radii_lp(x, y):
    """
    Solve for maximum radii given fixed centers using Linear Programming.
    Maximize sum(r) subject to r_i + r_j <= d_ij and r_i <= dist_to_bound.
    This is a convex problem, so the solution is optimal.
    """
    c = -np.ones(N)
    A_ub = []
    b_ub = []
    
    # Pairwise constraints: r_i + r_j <= d_ij
    for i in range(N):
        for j in range(i+1, N):
            d = np.hypot(x[i]-x[j], y[i]-y[j])
            row = np.zeros(N)
            row[i] = 1.0
            row[j] = 1.0
            A_ub.append(row)
            b_ub.append(d)
            
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
        
    # Fallback: iterative shrink if LP fails
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

def _optimize_centers_lp_guided(x0, start_time, time_budget=100):
    """
    Optimize centers (x, y) using L-BFGS-B.
    The objective function uses the LP solver to compute radii.
    Since the LP solution is non-smooth, we use a soft approximation or 
    simply rely on the fact that L-BFGS-B can handle piecewise linear functions 
    reasonably well if we don't get stuck in sharp corners, or we use 
    Nelder-Mead for robustness. 
    However, calling LP inside the objective is slow. 
    Strategy: 
    1. Use the standard penalty objective (fast) to get a good center layout.
    2. Use the LP solver to compute the exact radii for that layout.
    3. Do a local polish on centers using the LP-computed radii as fixed constraints? 
       No, that's not quite right.
    
    Revised Strategy for this function:
    We will run the standard penalty optimization on (x, y, r).
    Then, we will take the best (x, y) and compute r via LP.
    Then, we will do a final "center-only" optimization where we maximize 
    the sum of LP radii. Since this is non-smooth, we will use Nelder-Mead 
    or a very careful L-BFGS-B with a smoothed LP objective (e.g., using 
    the dual variables or just the primal values with a small noise).
    
    Actually, the most robust "structural" improvement is simply:
    1. Run multi-start penalty optimization on (x, y, r).
    2. For each best candidate, re-compute r via LP.
    3. Keep the best (x, y, r_LP) combination.
    4. Perform a final local search on (x, y) using the LP radii as the objective 
       (using Nelder-Mead to handle non-smoothness).
    """
    # This function is a placeholder for the actual logic in construct()
    pass

def _nelder_mead_center_polish(x0, y0, start_time, time_budget=10):
    """
    Polish centers (x, y) using Nelder-Mead to maximize sum of LP radii.
    """
    def obj(params):
        if _elapsed(start_time) > TIME_LIMIT - 2:
            return 1e10
        x = params[:N]
        y = params[N:2*N]
        # Clip to valid region to avoid LP failure
        x = np.clip(x, 0.01, 0.99)
        y = np.clip(y, 0.01, 0.99)
        r = _optimize_radii_lp(x, y)
        return -np.sum(r)
    
    x0 = x0.copy()
    y0 = y0.copy()
    x0 = np.clip(x0, 0.01, 0.99)
    y0 = np.clip(y0, 0.01, 0.99)
    params0 = np.concatenate([x0, y0])
    
    try:
        res = minimize(obj, params0, method='Nelder-Mead', 
                       options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8})
        x_opt = res.x[:N]
        y_opt = res.x[N:2*N]
        r_opt = _optimize_radii_lp(x_opt, y_opt)
        return x_opt, y_opt, r_opt
    except Exception:
        r_opt = _optimize_radii_lp(x0, y0)
        return x0, y0, r_opt

def construct():
    start_time = time.time()
    
    # Generate diverse initializations
    inits = []
    
    # Hexagonal
    hx, hy, hr = _hex_init()
    inits.append(np.concatenate([hx, hy, hr]))
    
    # Rectangular
    rx, ry, rr = _rect_init()
    inits.append(np.concatenate([rx, ry, rr]))
    
    # Multiple random inits
    for seed in [42, 123, 456, 789, 101, 202, 303, 404, 505, 606]:
        rng = np.random.RandomState(seed)
        rx2 = rng.uniform(0.1, 0.9, N)
        ry2 = rng.uniform(0.1, 0.9, N)
        rr2 = np.full(N, 0.06)
        inits.append(np.concatenate([rx2, ry2, rr2]))
        
    # Perturbed hex
    rng = np.random.RandomState(7)
    for pert in [0.03, 0.05, 0.08]:
        hx2, hy2, hr2 = _hex_init()
        hx2 += rng.uniform(-pert, pert, N)
        hy2 += rng.uniform(-pert, pert, N)
        hx2 = np.clip(hx2, 0, 1)
        hy2 = np.clip(hy2, 0, 1)
        inits.append(np.concatenate([hx2, hy2, hr2]))
    
    best_params = None
    best_score = -np.inf
    best_x = None
    best_y = None
    
    # Phase 1: Penalty-based optimization on (x, y, r)
    for init_idx, x0 in enumerate(inits):
        if _elapsed(start_time) > TIME_LIMIT - 30:
            break
        
        x0a = x0[:N].copy()
        x0b = x0[N:2*N].copy()
        x0c = x0[2*N:3*N].copy()
        
        # Push apart to resolve initial overlaps
        x0a, x0b = _push_apart(x0a, x0b, x0c, iterations=30)
        # Re-compute radii based on pushed positions using LP for better start
        x0c = _optimize_radii_lp(x0a, x0b)
        
        x0_opt = np.concatenate([x0a, x0b, x0c])
        
        try:
            res = minimize(_objective, x0_opt, method='L-BFGS-B',
                           args=(start_time,),
                           options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-10,
                                    'maxfun': 30000})
            result = res.x
        except Exception:
            result = x0_opt
            
        vx = result[:N]
        vy = result[N:2*N]
        vr = result[2*N:3*N]
        
        # Make valid
        vx, vy, vr = _make_valid(vx, vy, vr)
        
        # **Structural Change**: Use LP to compute optimal radii for these centers
        vr_lp = _optimize_radii_lp(vx, vy)
        
        # Check if LP radii are valid
        if _is_valid(vx, vy, vr_lp):
            score = _sum_r(vr_lp)
            if score > best_score:
                best_score = score
                best_params = np.concatenate([vx, vy, vr_lp])
                best_x = vx.copy()
                best_y = vy.copy()
        else:
            # Fallback to relaxed radii if LP failed or invalid
            score = _sum_r(vr)
            if score > best_score:
                best_score = score
                best_params = np.concatenate([vx, vy, vr])
                best_x = vx.copy()
                best_y = vy.copy()
                
    # Phase 2: Nelder-Mead Polish on Centers using LP Objective
    # This allows us to escape local minima of the penalty method by using 
    # the exact LP objective in a derivative-free manner.
    if best_params is not None and _elapsed(start_time) < TIME_LIMIT - 15:
        # Try polishing the best solution
        px, py, pr = _nelder_mead_center_polish(best_x, best_y, start_time, time_budget=10)
        if _is_valid(px, py, pr):
            p_score = _sum_r(pr)
            if p_score > best_score:
                best_score = p_score
                best_params = np.concatenate([px, py, pr])
                best_x = px.copy()
                best_y = py.copy()
                
        # Try polishing a few other top candidates if we kept them? 
        # We only kept the best one. Let's try polishing the initial hex/rect with LP objective directly?
        # This might be expensive. Let's just do one more polish on the best if time permits.
        
    # Final safety check and return
    if best_params is None:
        x, y, r = _rect_init()
        best_params = np.concatenate([x, y, r])
    
    x = best_params[:N]
    y = best_params[N:2*N]
    r = best_params[2*N:3*N]
    
    # Final LP pass to ensure maximum radii for the final geometry
    r_final = _optimize_radii_lp(x, y)
    if _is_valid(x, y, r_final):
        r = r_final
    
    result = np.column_stack((x, y, r))
    return result
