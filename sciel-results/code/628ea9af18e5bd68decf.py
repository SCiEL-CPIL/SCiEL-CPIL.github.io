import numpy as np
from scipy.optimize import minimize, linprog
from scipy.spatial import Delaunay
import time

N = 21

def _softplus(x, beta=50.0):
    out = np.zeros_like(x)
    arg = beta * x
    large = arg > 50
    small = arg < -50
    mid = ~(large | small)
    out[large] = x[large]
    out[mid] = (1.0 / beta) * np.log1p(np.exp(arg[mid]))
    return out

def _delaunay_edges(points):
    """Return edge indices from Delaunay triangulation."""
    try:
        tri = Delaunay(points)
        edges = set()
        for simplex in tri.simplices:
            for i in range(3):
                for j in range(i+1, 3):
                    a, b = simplex[i], simplex[j]
                    if a > b:
                        a, b = b, a
                    edges.add((a, b))
        return list(edges)
    except Exception:
        # Fallback: all pairs
        return [(i, j) for i in range(N) for j in range(i+1, N)]

def _solve_lp_radii(xs, ys, w, h, edges=None):
    """
    Solve for maximum radii given fixed positions.
    Maximize sum(r) subject to:
      r_i + r_j <= dist_ij for all edges (i,j)
      r_i <= x_i, r_i <= w-x_i, r_i <= y_i, r_i <= h-y_i
      r_i >= 0
    """
    if edges is None:
        edges = _delaunay_edges(np.column_stack([xs, ys]))
    
    n = N
    # Variables: r_0, ..., r_{n-1}
    # Objective: maximize sum(r) => minimize -sum(r)
    c = -np.ones(n)
    
    # Inequality constraints: A_ub @ r <= b_ub
    A_ub = []
    b_ub = []
    
    # Pairwise constraints from edges
    for i, j in edges:
        dx = xs[i] - xs[j]
        dy = ys[i] - ys[j]
        dist = np.sqrt(dx*dx + dy*dy)
        row = np.zeros(n)
        row[i] = 1
        row[j] = 1
        A_ub.append(row)
        b_ub.append(dist)
    
    # Boundary constraints
    for i in range(n):
        for bound_val in [xs[i], w-xs[i], ys[i], h-ys[i]]:
            row = np.zeros(n)
            row[i] = 1
            A_ub.append(row)
            b_ub.append(bound_val)
    
    # Also add all-pair constraints as safety (in case Delaunay misses close pairs)
    # But only for pairs that are very close (within sum of current radii estimate)
    # Actually, let's add all pairs to be safe since N=21 is small
    # Remove duplicates from edges
    edge_set = set(edges)
    for i in range(n):
        for j in range(i+1, n):
            if (i, j) not in edge_set:
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                dist = np.sqrt(dx*dx + dy*dy)
                row = np.zeros(n)
                row[i] = 1
                row[j] = 1
                A_ub.append(row)
                b_ub.append(dist)
    
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    
    bounds = [(0, None)] * n
    
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            return res.x
    except Exception:
        pass
    
    # Fallback: greedy radius assignment
    rs = np.zeros(n)
    for i in range(n):
        max_r = min(xs[i], w-xs[i], ys[i], h-ys[i])
        for j in range(n):
            if i != j:
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                dist = np.sqrt(dx*dx + dy*dy)
                max_r = min(max_r, dist - rs[j])
        rs[i] = max(0, max_r)
    return rs

def _is_valid(params, w, h, tol=1e-6):
    xs = params[0:N]
    ys = params[N:2*N]
    rs = params[2*N:3*N]
    
    if np.any(rs < -tol): return False
    if np.any(xs < -tol) or np.any(w - xs < -tol): return False
    if np.any(ys < -tol) or np.any(h - ys < -tol): return False
    if np.any(rs - xs < -tol) or np.any(rs - (w - xs) < -tol): return False
    if np.any(rs - ys < -tol) or np.any(rs - (h - ys) < -tol): return False
    
    for i in range(N):
        for j in range(i+1, N):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist = np.sqrt(dx*dx + dy*dy)
            if dist < rs[i] + rs[j] - tol:
                return False
    return True

def _make_hex_init(w, h, rng, pattern=None):
    if pattern is None:
        patterns = [
            [5, 4, 5, 4, 3],
            [4, 5, 4, 5, 3],
            [5, 5, 5, 4, 2],
            [3, 4, 5, 4, 5],
            [4, 4, 5, 4, 4],
            [7, 7, 7],
            [4, 4, 4, 4, 5],
            [6, 5, 5, 5],
            [5, 6, 5, 5],
            [7, 6, 8],
        ]
        pattern = patterns[rng.integers(len(patterns))]
    
    xs = np.zeros(N)
    ys = np.zeros(N)
    rs = np.zeros(N)
    
    idx = 0
    n_rows = len(pattern)
    row_h = h / n_rows
    
    for row_idx, n_cols in enumerate(pattern):
        col_w = w / n_cols
        offset = 0.0 if row_idx % 2 == 0 else col_w / 2
        for col_idx in range(n_cols):
            if idx >= N: break
            x = (col_idx + 0.5) * col_w + offset
            y = (row_idx + 0.5) * row_h
            # Clamp to bounds
            x = np.clip(x, 0, w)
            y = np.clip(y, 0, h)
            xs[idx] = x
            ys[idx] = y
            r_est = 0.5 * min(x, w - x, y, h - y)
            rs[idx] = max(0.01, r_est)
            idx += 1
    
    if idx != N:
        return None
    
    xs += rng.normal(0, 0.01, N)
    ys += rng.normal(0, 0.01, N)
    rs = np.clip(rs, 0.01, 0.5)
    
    return np.concatenate([xs, ys, rs])

def _make_grid_init(w, h, rng, noise_scale=0.02):
    best_init = None
    best_sum = -1
    for rows in range(3, 8):
        cols = int(np.ceil(N / rows))
        if cols < 2: continue
        
        xs = np.zeros(N)
        ys = np.zeros(N)
        rs = np.zeros(N)
        
        for i in range(N):
            row = i // cols
            col = i % cols
            xs[i] = (col + 0.5) * (w / cols)
            ys[i] = (row + 0.5) * (h / rows)
            r_est = 0.5 * min(xs[i], w - xs[i], ys[i], h - ys[i])
            rs[i] = max(0.01, r_est)
        
        xs += rng.normal(0, noise_scale, N)
        ys += rng.normal(0, noise_scale, N)
        rs = np.clip(rs, 0.01, 0.5)
        
        params = np.concatenate([xs, ys, rs])
        s = np.sum(rs)
        if s > best_sum:
            best_sum = s
            best_init = params
    return best_init

def _make_random_init(w, h, rng):
    xs = rng.uniform(0.1, w - 0.1, N)
    ys = rng.uniform(0.1, h - 0.1, N)
    rs = rng.uniform(0.05, 0.2, N)
    return np.concatenate([xs, ys, rs])

def _polish_positions(params, w, h, rng):
    """
    Optimize positions (x, y) while using LP to compute optimal radii.
    This decouples position optimization from radius optimization.
    """
    xs = params[0:N].copy()
    ys = params[N:2*N].copy()
    
    def objective_pos(pos):
        x = pos[0:N]
        y = pos[N:2*N]
        
        # Compute radii via LP
        rs = _solve_lp_radii(x, y, w, h)
        
        # Penalty for any numerical issues
        if np.any(rs < -1e-10):
            return 1e10 + np.sum(rs)
        
        # Return negative sum of radii (we want to maximize)
        return -np.sum(rs)
    
    def grad_pos(pos):
        x = pos[0:N]
        y = pos[N:2*N]
        
        # Numerical gradient
        eps = 1e-7
        grad = np.zeros(2*N)
        for i in range(2*N):
            pos_plus = pos.copy()
            pos_plus[i] += eps
            f_plus = objective_pos(pos_plus)
            grad[i] = (f_plus - objective_pos(pos)) / eps
        return grad
    
    # Use L-BFGS-B for position optimization
    bounds = [(0, w)] * N + [(0, h)] * N
    
    try:
        res = minimize(objective_pos, params[0:2*N], 
                      method='L-BFGS-B',
                      bounds=bounds,
                      options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8})
        
        if res.success:
            new_xs = res.x[0:N]
            new_ys = res.x[N:2*N]
            new_rs = _solve_lp_radii(new_xs, new_ys, w, h)
            return np.concatenate([new_xs, new_ys, new_rs])
    except Exception:
        pass
    
    return params

def _force_directed(params, w, h, n_iter=500):
    """
    Force-directed relaxation: push apart overlapping circles, pull toward center.
    """
    xs = params[0:N].copy()
    ys = params[N:2*N].copy()
    rs = params[2*N:3*N].copy()
    
    # Initial radii from LP
    rs = _solve_lp_radii(xs, ys, w, h)
    
    for it in range(n_iter):
        # Compute forces
        fx = np.zeros(N)
        fy = np.zeros(N)
        
        # Repulsion between overlapping circles
        for i in range(N):
            for j in range(i+1, N):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                dist = np.sqrt(dx*dx + dy*dy)
                min_dist = rs[i] + rs[j]
                
                if dist < min_dist and dist > 1e-10:
                    overlap = min_dist - dist
                    force = 0.1 * overlap
                    fx[i] += force * dx / dist
                    fy[i] += force * dy / dist
                    fx[j] -= force * dx / dist
                    fy[j] -= force * dy / dist
                elif dist <= 1e-10:
                    # Random nudge
                    fx[i] += 0.01 * rng.normal()
                    fy[i] += 0.01 * rng.normal()
                    fx[j] -= 0.01 * rng.normal()
                    fy[j] -= 0.01 * rng.normal()
        
        # Attraction to center (weak)
        for i in range(N):
            fx[i] += 0.001 * (w/2 - xs[i])
            fy[i] += 0.001 * (h/2 - ys[i])
        
        # Boundary forces
        for i in range(N):
            if xs[i] < rs[i]:
                fx[i] += 0.5 * (rs[i] - xs[i])
            if xs[i] > w - rs[i]:
                fx[i] -= 0.5 * (xs[i] - (w - rs[i]))
            if ys[i] < rs[i]:
                fy[i] += 0.5 * (rs[i] - ys[i])
            if ys[i] > h - rs[i]:
                fy[i] -= 0.5 * (ys[i] - (h - rs[i]))
        
        # Update positions
        xs += 0.1 * fx
        ys += 0.1 * fy
        
        # Clamp to bounds
        xs = np.clip(xs, 0, w)
        ys = np.clip(ys, 0, h)
        
        # Recompute radii via LP
        rs = _solve_lp_radii(xs, ys, w, h)
    
    return np.concatenate([xs, ys, rs])

def construct():
    start_time = time.time()
    time_limit = 145.0  # Safety margin
    
    best_val = 0.0
    best_params = None
    best_w = 1.0
    
    rng = np.random.default_rng(42)
    
    # Search over aspect ratios
    w_values = np.linspace(0.6, 1.4, 17)
    
    for w in w_values:
        if time.time() - start_time > time_limit - 30:
            break
            
        h = 2.0 - w
        
        # Multiple initialization strategies
        inits = []
        
        # Hex patterns
        for _ in range(3):
            init = _make_hex_init(w, h, rng)
            if init is not None:
                inits.append(init)
        
        # Grid
        init = _make_grid_init(w, h, rng, noise_scale=0.02)
        if init is not None:
            inits.append(init)
        
        # Random
        inits.append(_make_random_init(w, h, rng))
        
        for params0 in inits:
            if time.time() - start_time > time_limit - 15:
                break
            
            # Step 1: Force-directed relaxation
            try:
                params_fd = _force_directed(params0, w, h, n_iter=300)
            except Exception:
                params_fd = params0
            
            # Step 2: Position optimization with LP radii
            try:
                params_opt = _polish_positions(params_fd, w, h, rng)
            except Exception:
                params_opt = params_fd
            
            # Step 3: Final LP solve for optimal radii
            xs = params_opt[0:N]
            ys = params_opt[N:2*N]
            rs = _solve_lp_radii(xs, ys, w, h)
            params_final = np.concatenate([xs, ys, rs])
            
            # Validate
            if _is_valid(params_final, w, h):
                val = np.sum(rs)
                if val > best_val:
                    best_val = val
                    best_params = params_final.copy()
                    best_w = w
    
    # Final polish: try to improve best solution
    if best_params is not None and time.time() - start_time < time_limit - 10:
        w = best_w
        h = 2.0 - w
        
        # Try force-directed with more iterations
        try:
            params_fd = _force_directed(best_params, w, h, n_iter=500)
            xs = params_fd[0:N]
            ys = params_fd[N:2*N]
            rs = _solve_lp_radii(xs, ys, w, h)
            params_final = np.concatenate([xs, ys, rs])
            
            if _is_valid(params_final, w, h):
                val = np.sum(rs)
                if val > best_val:
                    best_val = val
                    best_params = params_final.copy()
        except Exception:
            pass
        
        # Try position optimization again
        try:
            params_opt = _polish_positions(best_params, w, h, rng)
            xs = params_opt[0:N]
            ys = params_opt[N:2*N]
            rs = _solve_lp_radii(xs, ys, w, h)
            params_final = np.concatenate([xs, ys, rs])
            
            if _is_valid(params_final, w, h):
                val = np.sum(rs)
                if val > best_val:
                    best_val = val
                    best_params = params_final.copy()
        except Exception:
            pass
    
    if best_params is not None:
        out = best_params.reshape(N, 3)
        out[:, 2] = np.maximum(0, out[:, 2])
        return out
    else:
        # Fallback: simple grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
