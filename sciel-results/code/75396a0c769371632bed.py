"""Circle packing in unit square using smooth penalty-based optimization.

Structural change: optimize all 78 variables (x, y, r) jointly with a
differentiable smooth surrogate objective that penalizes constraint violations.
This avoids the non-differentiable Delaunay-based radius derivation and allows
gradient-based methods (L-BFGS-B) to work effectively on the full parameter space.

Pipeline:
1. Multiple diverse initializations
2. L-BFGS-B on smooth penalty objective (all 78 vars)
3. Discrete best-fit local search for refinement
4. Final high-precision polish
- Deadline guard for safety
"""
import time
import numpy as np
from scipy.optimize import minimize

N = 26
DEADLINE = 145.0  # Leave 5s safety margin under 150s
START = time.time()


def _elapsed():
    return time.time() - START


def _unpack(params):
    """Unpack 78-dim vector into x, y, r arrays."""
    x = params[0::3]
    y = params[1::3]
    r = params[2::3]
    return x, y, r


def _smooth_objective(params):
    """
    Smooth penalty objective to minimize: -sum(r) + penalty for violations.
    Uses soft penalties that are differentiable.
    """
    x, y, r = _unpack(params)
    
    # Boundary constraint: r_i <= min(x_i, 1-x_i, y_i, 1-y_i)
    d_bound = np.minimum(np.minimum(x, 1.0 - x), np.minimum(y, 1.0 - y))
    bound_viol = np.maximum(0.0, r - d_bound)
    
    # Pairwise constraint: r_i + r_j <= dist(i,j)
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist = np.sqrt(dx * dx + dy * dy)
    need = r[:, None] + r[None, :]
    np.fill_diagonal(need, 0.0)
    np.fill_diagonal(dist, np.inf)
    pair_viol = np.maximum(0.0, need - dist)
    
    # Soft penalty: quadratic for small violations, linear for large
    bound_penalty = np.sum(bound_viol ** 2) + 10.0 * np.sum(bound_viol)
    pair_penalty = np.sum(pair_viol ** 2) + 10.0 * np.sum(pair_viol)
    
    return -np.sum(r) + 1000.0 * (bound_penalty + pair_penalty)


def _smooth_gradient(params):
    """
    Analytic gradient of the smooth penalty objective.
    """
    x, y, r = _unpack(params)
    n = N
    
    # Compute distances
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist = np.sqrt(dx * dx + dy * dy + 1e-30)
    np.fill_diagonal(dist, 1.0)  # avoid div by zero
    np.fill_diagonal(dist, np.inf)
    
    # Boundary distances
    d_bound = np.minimum(np.minimum(x, 1.0 - x), np.minimum(y, 1.0 - y))
    
    # Gradients
    gx = np.zeros(n)
    gy = np.zeros(n)
    gr = np.zeros(n)
    
    # Objective: -sum(r)
    gr -= 1.0
    
    # Boundary penalty: sum(max(0, r_i - d_bound_i)^2 + 10*max(0, r_i - d_bound_i))
    for i in range(n):
        v = r[i] - d_bound[i]
        if v > 0:
            # d/dx of d_bound depends on which constraint is active
            # d_bound = min(x, 1-x, y, 1-y)
            # We use subgradient: pick the active constraint
            if x[i] <= 1.0 - x[i] and x[i] <= y[i] and x[i] <= 1.0 - y[i]:
                gx[i] += 1000.0 * (2.0 * v + 10.0) * (-1.0)
            elif 1.0 - x[i] <= x[i] and 1.0 - x[i] <= y[i] and 1.0 - x[i] <= 1.0 - y[i]:
                gx[i] += 1000.0 * (2.0 * v + 10.0) * (1.0)
            elif y[i] <= x[i] and y[i] <= 1.0 - x[i] and y[i] <= 1.0 - y[i]:
                gy[i] += 1000.0 * (2.0 * v + 10.0) * (-1.0)
            else:
                gy[i] += 1000.0 * (2.0 * v + 10.0) * (1.0)
            gr[i] += 1000.0 * (2.0 * v + 10.0)
    
    # Pairwise penalty: sum_{i<j} max(0, r_i+r_j - d_ij)^2 + 10*max(0, r_i+r_j - d_ij)
    for i in range(n):
        for j in range(i + 1, n):
            v = r[i] + r[j] - dist[i, j]
            if v > 0:
                coef = 1000.0 * (2.0 * v + 10.0)
                # d/dx_i of dist = (x_i - x_j)/dist
                gx[i] += coef * (x[i] - x[j]) / dist[i, j]
                gx[j] += coef * (x[j] - x[i]) / dist[i, j]
                gy[i] += coef * (y[i] - y[j]) / dist[i, j]
                gy[j] += coef * (y[j] - y[i]) / dist[i, j]
                gr[i] += coef
                gr[j] += coef
    
    # Pack gradient
    g = np.zeros(3 * n)
    g[0::3] = gx
    g[1::3] = gy
    g[2::3] = gr
    return g


def _is_valid(x, y, r, eps=1e-5):
    """Check if configuration is valid."""
    d = np.minimum(np.minimum(x, 1.0 - x), np.minimum(y, 1.0 - y))
    if np.any(r + eps > d):
        return False
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist = np.sqrt(dx * dx + dy * dy)
    need = r[:, None] + r[None, :]
    np.fill_diagonal(need, 0.0)
    np.fill_diagonal(dist, np.inf)
    if np.any(need + eps > dist):
        return False
    return True


def _fix_to_valid(x, y, r, max_iters=100):
    """
    Iteratively shrink radii and adjust positions to eliminate overlaps.
    Returns valid (x, y, r).
    """
    x = np.clip(x, 0.0, 1.0).copy()
    y = np.clip(y, 0.0, 1.0).copy()
    r = np.maximum(r, 1e-8).copy()
    
    for _ in range(max_iters):
        # Boundary constraint
        d_bound = np.minimum(np.minimum(x, 1.0 - x), np.minimum(y, 1.0 - y))
        r = np.minimum(r, np.maximum(d_bound, 1e-8))
        
        # Pairwise constraints
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist = np.sqrt(dx * dx + dy * dy)
        np.fill_diagonal(dist, np.inf)
        
        changed = False
        for i in range(N):
            for j in range(i + 1, N):
                if dist[i, j] < r[i] + r[j] - 1e-10:
                    total_r = r[i] + r[j]
                    if total_r > 0:
                        scale = dist[i, j] / total_r * 0.999
                        r[i] *= scale
                        r[j] *= scale
                    else:
                        r[i] = r[j] = 1e-6
                    changed = True
        
        if not changed:
            break
    
    return x, y, np.maximum(r, 1e-6)


def _best_fit_local_search(x, y, r, n_steps=2000, step_sizes=None):
    """
    Discrete best-fit local search: try moving each circle in small steps
    and accept if it improves sum(r) while maintaining validity.
    """
    if step_sizes is None:
        step_sizes = [0.01, 0.005, 0.002, 0.001]
    
    best_val = float(np.sum(r))
    best_x, best_y, best_r = x.copy(), y.copy(), r.copy()
    
    directions = np.array([[1, 0], [-1, 0], [0, 1], [0, -1],
                           [0.707, 0.707], [-0.707, 0.707],
                           [0.707, -0.707], [-0.707, -0.707]])
    
    for step in step_sizes:
        improved = True
        while improved and _elapsed() < DEADLINE - 5.0:
            improved = False
            for i in range(N):
                if _elapsed() > DEADLINE - 5.0:
                    return best_x, best_y, best_r
                for dx, dy in directions:
                    nx = np.clip(best_x[i] + dx * step, 0.0, 1.0)
                    ny = np.clip(best_y[i] + dy * step, 0.0, 1.0)
                    if abs(nx - best_x[i]) < 1e-10 and abs(ny - best_y[i]) < 1e-10:
                        continue
                    
                    trial_x = best_x.copy()
                    trial_y = best_y.copy()
                    trial_x[i] = nx
                    trial_y[i] = ny
                    trial_r = best_r.copy()
                    
                    # Recompute max possible radius for circle i
                    d_bound = min(nx, 1.0 - nx, ny, 1.0 - ny)
                    min_dist = np.inf
                    for j in range(N):
                        if j == i:
                            continue
                        d = np.sqrt((trial_x[i] - trial_x[j])**2 + (trial_y[i] - trial_y[j])**2)
                        min_dist = min(min_dist, d - trial_r[j])
                    max_r = max(1e-6, 0.5 * min(d_bound, min_dist) * 2.0)
                    # Actually max_r = min(d_bound, min_dist)
                    max_r = min(d_bound, min_dist)
                    max_r = max(max_r, 1e-6)
                    
                    new_r_i = max_r
                    trial_r[i] = new_r_i
                    
                    if _is_valid(trial_x, trial_y, trial_r, eps=1e-6):
                        new_val = float(np.sum(trial_r))
                        if new_val > best_val + 1e-10:
                            best_val = new_val
                            best_x, best_y, best_r = trial_x, trial_y, trial_r
                            improved = True
    
    return best_x, best_y, best_r


def _grid_init(rng):
    """5x5 grid + 1 center."""
    cols, rows = 5, 5
    xs = (np.arange(cols) + 0.5) / cols
    ys = (np.arange(rows) + 0.5) / rows
    pts = [(xs[i], ys[j]) for j in range(rows) for i in range(cols)]
    extra = [(0.5, 0.5)]
    pts = pts + extra[:N - len(pts)]
    x = np.array([p[0] for p in pts[:N]])
    y = np.array([p[1] for p in pts[:N]])
    r = np.full(N, 0.1)
    return x, y, r


def _hex_init(rng):
    """Hexagonal lattice."""
    x = np.zeros(N)
    y = np.zeros(N)
    spacing = 0.18
    for i in range(N):
        row = i // 6
        col = i % 6
        x[i] = 0.12 + col * spacing + (0.5 * spacing if row % 2 else 0)
        y[i] = 0.12 + row * spacing * 0.866
        if x[i] > 0.9:
            x[i] = 0.9
        if y[i] > 0.9:
            y[i] = 0.9
    r = np.full(N, 0.08)
    return x, y, r


def _random_init(rng):
    x = rng.uniform(0.1, 0.9, size=N)
    y = rng.uniform(0.1, 0.9, size=N)
    r = np.full(N, 0.05)
    return x, y, r


def _optimize_smooth(x, y, r, maxiter=2000):
    """Optimize all 78 variables using L-BFGS-B with analytic gradient."""
    params = np.zeros(3 * N)
    params[0::3] = x
    params[1::3] = y
    params[2::3] = r
    
    bounds = []
    for i in range(N):
        bounds.append((0.0, 1.0))  # x
        bounds.append((0.0, 1.0))  # y
        bounds.append((1e-6, 0.5))  # r
    
    try:
        res = minimize(_smooth_objective, params, jac=_smooth_gradient,
                       method='L-BFGS-B', bounds=bounds,
                       options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-8,
                                'maxfun': maxiter * 3})
    except Exception:
        return x, y, r
    
    v = res.x
    xx = np.clip(v[0::3], 0.0, 1.0)
    yy = np.clip(v[1::3], 0.0, 1.0)
    rr = np.clip(v[2::3], 1e-6, 0.5)
    
    return xx, yy, rr


def construct():
    best = None
    best_val = -np.inf
    rng = np.random.default_rng(42)

    inits = [_grid_init, _hex_init, _random_init]

    for init_func in inits:
        if _elapsed() > DEADLINE - 20.0:
            break
        
        base_x, base_y, base_r = init_func(rng)
        
        for restart in range(8):
            if _elapsed() > DEADLINE - 20.0:
                break
            
            # Add jitter
            x0 = np.clip(base_x + rng.normal(0, 0.03, N), 0.01, 0.99)
            y0 = np.clip(base_y + rng.normal(0, 0.03, N), 0.01, 0.99)
            r0 = np.clip(base_r + rng.normal(0, 0.01, N), 0.01, 0.4)
            
            # Smooth gradient optimization
            xx, yy, rr = _optimize_smooth(x0, y0, r0, maxiter=1000)
            
            # Fix to valid
            xx, yy, rr = _fix_to_valid(xx, yy, rr)
            
            if _valid := _is_valid(xx, yy, rr):
                val = float(np.sum(rr))
                if val > best_val:
                    best_val = val
                    best = np.column_stack([xx, yy, rr])
            
            # Best-fit local search refinement
            if _elapsed() < DEADLINE - 10.0:
                xx2, yy2, rr2 = _best_fit_local_search(xx, yy, rr, 
                                                        step_sizes=[0.02, 0.01, 0.005, 0.002])
                if _is_valid(xx2, yy2, rr2):
                    val2 = float(np.sum(rr2))
                    if val2 > best_val:
                        best_val = val2
                        best = np.column_stack([xx2, yy2, rr2])
            
            # Additional smooth polish on best result
            if _elapsed() < DEADLINE - 5.0:
                xx3, yy3, rr3 = _optimize_smooth(xx2, yy2, rr2, maxiter=500)
                xx3, yy3, rr3 = _fix_to_valid(xx3, yy3, rr3)
                if _is_valid(xx3, yy3, rr3):
                    val3 = float(np.sum(rr3))
                    if val3 > best_val:
                        best_val = val3
                        best = np.column_stack([xx3, yy3, rr3])

    if best is not None:
        return best

    # Guaranteed fallback
    side = int(np.ceil(np.sqrt(N)))
    cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
             for i in range(side) for j in range(side)]
    cells = cells[:N]
    while len(cells) < N:
        cells.append((0.5 / side, 0.5 / side, 0.1 / side))
    return np.array(cells[:N])
