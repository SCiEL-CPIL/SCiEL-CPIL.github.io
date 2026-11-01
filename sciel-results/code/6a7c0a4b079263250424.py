import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import Delaunay

N = 50
_START_TIME = None
_DEADLINE_MARGIN = 15.0  # seconds before the 150s limit

def _compute_radius(x):
    """Compute the actual radius (score) for a configuration."""
    x = x.reshape(N, 2)
    dx = x[:, 0:1] - x[:, 0:1].T
    dy = x[:, 1:2] - x[:, 1:2].T
    dists_sq = dx**2 + dy**2
    np.fill_diagonal(dists_sq, 1.0)
    min_dist = np.sqrt(np.min(dists_sq))
    
    wx = 0.5 - np.abs(x[:, 0])
    wy = 0.5 - np.abs(x[:, 1])
    min_wall = np.min(np.minimum(wx, wy))
    
    return min(min_dist / 2.0, min_wall)

def _softmin_obj_and_grad(x, beta):
    """
    Soft-min objective: maximize min(pair distances, 2*wall distances).
    We minimize the negative of this soft-min.
    """
    x = x.reshape(N, 2)
    # Compute pair distances
    dx = x[:, 0:1] - x[:, 0:1].T
    dy = x[:, 1:2] - x[:, 1:2].T
    dists_sq = dx**2 + dy**2
    np.fill_diagonal(dists_sq, 1.0)
    dists = np.sqrt(dists_sq)
    
    # Wall distances
    wx = 0.5 - np.abs(x[:, 0])
    wy = 0.5 - np.abs(x[:, 1])
    walls = np.minimum(wx, wy)
    
    # Values for softmin: pair distances and 2*wall distances
    iu = np.triu_indices(N, k=1)
    pair_dists = dists[iu]
    wall_vals = 2 * walls
    
    vals = np.concatenate([pair_dists, wall_vals])
    
    # Softmin calculation
    max_val = np.max(vals)
    exp_vals = np.exp(-beta * (vals - max_val))
    sum_exp = np.sum(exp_vals)
    softmin_val = max_val - (1.0 / beta) * np.log(sum_exp)
    
    obj_val = -softmin_val
    
    # Gradient
    p = exp_vals / sum_exp
    
    n_pairs = len(pair_dists)
    pair_p = p[:n_pairs]
    wall_p = p[n_pairs:]
    
    g_soft = np.zeros_like(x)
    
    # Pairs
    for idx in range(n_pairs):
        p_k = pair_p[idx]
        if p_k < 1e-15:
            continue
        i, j = iu[0][idx], iu[1][idx]
        d = pair_dists[idx]
        if d < 1e-8:
            continue
        diff = x[i] - x[j]
        g_soft[i] += p_k * (diff / d)
        g_soft[j] -= p_k * (diff / d)
        
    # Walls
    for i in range(N):
        p_k = wall_p[i]
        if p_k < 1e-15:
            continue
        wx_i = wx[i]
        wy_i = wy[i]
        if wx_i < wy_i:
            g_soft[i, 0] += p_k * 2.0 * (-np.sign(x[i, 0]))
        elif wy_i < wx_i:
            g_soft[i, 1] += p_k * 2.0 * (-np.sign(x[i, 1]))
        else:
            g_soft[i, 0] += p_k * 2.0 * (-np.sign(x[i, 0])) * 0.5
            g_soft[i, 1] += p_k * 2.0 * (-np.sign(x[i, 1])) * 0.5
            
    grad_obj = -g_soft
    
    return obj_val, grad_obj

def _init_hex(r_guess=0.07):
    """Initialize points in a hexagonal pattern."""
    dx = 2 * r_guess
    dy = np.sqrt(3) * r_guess
    
    points = []
    y = -0.5 + dy/2
    row_idx = 0
    while y <= 0.5:
        x_start = -0.5 + dx/2
        if row_idx % 2 == 1:
            x_start += dx/2
        x = x_start
        while x <= 0.5:
            points.append((x, y))
            x += dx
        y += dy
        row_idx += 1
        
    points = np.array(points)
    
    if len(points) >= N:
        idx = np.random.choice(len(points), N, replace=False)
        return points[idx]
    else:
        remaining = N - len(points)
        random_pts = np.random.uniform(-0.45, 0.45, (remaining, 2))
        return np.vstack([points, random_pts])

def _delaunay_contacts(x):
    """
    Given a configuration, compute the Delaunay triangulation and return
    the set of edges (contacts) that are near the minimum distance.
    """
    try:
        tri = Delaunay(x)
    except Exception:
        return None, None
    
    # Get unique edges from triangles
    edges = set()
    for simplex in tri.simplices:
        for i in range(3):
            for j in range(i+1, 3):
                a, b = simplex[i], simplex[j]
                if a > b:
                    a, b = b, a
                edges.add((a, b))
    
    # Compute distances for each edge
    edge_dists = {}
    for (i, j) in edges:
        d = np.linalg.norm(x[i] - x[j])
        edge_dists[(i, j)] = d
    
    # Also consider wall contacts
    wx = 0.5 - np.abs(x[:, 0])
    wy = 0.5 - np.abs(x[:, 1])
    wall_dists = np.minimum(wx, wy)
    
    # Find the minimum distance among all contacts (edges + walls)
    all_dists = list(edge_dists.values()) + [2*wd for wd in wall_dists]
    min_dist = min(all_dists)
    
    # Determine which contacts are "active" (within 1% of min)
    threshold = min_dist * 1.01
    active_edges = set()
    for (i, j), d in edge_dists.items():
        if d <= threshold:
            active_edges.add((i, j))
    
    active_walls = []
    for i in range(N):
        if 2 * wall_dists[i] <= threshold:
            active_walls.append(i)
    
    return active_edges, active_walls

def _solve_contact_system(x_init, active_edges, active_walls, r_target):
    """
    Given a set of contacts and a target radius, solve for positions.
    This is a least-squares problem: minimize the deviation from
    edge lengths = 2*r_target and wall distances = r_target.
    """
    n_vars = 2 * N
    n_eqs = len(active_edges) + len(active_walls)
    
    if n_eqs == 0:
        return None
    
    # Build linear system A x = b
    # Unknowns: x_0, y_0, x_1, y_1, ..., x_{N-1}, y_{N-1}
    # For each edge (i, j): |x_i - x_j|^2 = (2r)^2
    # For each wall i: min(0.5-|x_i|, 0.5-|y_i|) = r
    
    # We'll use a nonlinear solver on the residuals
    def residuals(pos):
        pos = pos.reshape(N, 2)
        res = []
        for (i, j) in active_edges:
            d = np.linalg.norm(pos[i] - pos[j])
            res.append(d - 2 * r_target)
        for i in active_walls:
            wx = 0.5 - np.abs(pos[i, 0])
            wy = 0.5 - np.abs(pos[i, 1])
            w = min(wx, wy)
            res.append(w - r_target)
        return np.array(res)
    
    try:
        from scipy.optimize import least_squares
        result = least_squares(residuals, x_init, method='lm', max_nfev=500)
        x_opt = result.x.reshape(N, 2)
        
        # Check if solution is valid
        if not np.all(np.isfinite(x_opt)):
            return None
        if np.any(x_opt < -0.5) or np.any(x_opt > 0.5):
            return None
        
        return x_opt
    except Exception:
        return None

def _delaunay_search(x_init, r_target, num_iterations=50, time_limit=10.0):
    """
    Iteratively refine the contact graph using Delaunay triangulation.
    """
    start_time = time.time()
    x = x_init.copy()
    
    for it in range(num_iterations):
        if time.time() - start_time > time_limit:
            break
        
        # Compute Delaunay contacts
        active_edges, active_walls = _delaunay_contacts(x)
        if active_edges is None:
            break
        
        # Solve for positions given the contact graph
        x_new = _solve_contact_system(x, active_edges, active_walls, r_target)
        if x_new is None:
            break
        
        # Check if the solution is feasible
        score = _compute_radius(x_new)
        if score < r_target * 0.99:
            # Solution is not good enough, try with slightly smaller r
            break
        
        x = x_new
    
    return x

def _binary_search_radius(x_init, time_budget=30.0):
    """
    Binary search on radius with feasibility checks.
    """
    start_time = time.time()
    
    # First, get a lower bound from the initial configuration
    r_low = _compute_radius(x_init)
    r_high = 0.075  # Upper bound
    
    # Binary search
    for _ in range(20):
        if time.time() - start_time > time_budget:
            break
        
        r_mid = (r_low + r_high) / 2
        
        # Try to find a feasible configuration for radius r_mid
        # Use the soft-min approach as a feasibility check
        beta = 100.0
        try:
            res = minimize(
                _softmin_obj_and_grad,
                x_init,
                args=(beta,),
                method='L-BFGS-B',
                jac=True,
                bounds=[(-0.5, 0.5)] * (2 * N),
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            x_opt = res.x.reshape(N, 2)
            score = _compute_radius(x_opt)
            
            if score >= r_mid:
                r_low = r_mid
                x_init = x_opt  # Use this as the next starting point
            else:
                r_high = r_mid
        except Exception:
            r_high = r_mid
    
    return r_low, x_init

def construct():
    global _START_TIME
    _START_TIME = time.time()
    deadline = _START_TIME + (150.0 - _DEADLINE_MARGIN)
    
    best_x = None
    best_score = -np.inf
    
    # Phase 1: Multiple restarts with soft-min optimization
    num_restarts = 15
    
    for k in range(num_restarts):
        elapsed = time.time() - _START_TIME
        if elapsed > deadline:
            break
        
        # Adaptive beta
        beta = 50.0 + 100.0 * (k / num_restarts)
        
        # Different initialization strategies
        if k == 0:
            x0 = _init_hex(0.07)
        elif k == 1:
            x0 = _init_hex(0.065)
        elif k == 2:
            x0 = _init_hex(0.06)
        elif k % 3 == 0:
            m = 7
            ticks = (np.arange(m) + 0.5) / m - 0.5
            grid = np.array([(x, y) for y in ticks for x in ticks])
            if len(grid) >= N:
                idx = np.random.choice(len(grid), N, replace=False)
                x0 = grid[idx]
            else:
                x0 = _init_hex(0.06)
        elif k % 3 == 1:
            x0 = np.random.uniform(-0.45, 0.45, (N, 2))
        else:
            x0 = _init_hex(0.065) + np.random.normal(0, 0.01, (N, 2))
            x0 = np.clip(x0, -0.48, 0.48)
        
        if not np.all(np.isfinite(x0)):
            continue
        
        # Optimization
        try:
            res = minimize(
                _softmin_obj_and_grad,
                x0,
                args=(beta,),
                method='L-BFGS-B',
                jac=True,
                bounds=[(-0.5, 0.5)] * (2 * N),
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            x_opt = res.x.reshape(N, 2)
            
            score = _compute_radius(x_opt)
            
            if score > best_score:
                best_score = score
                best_x = x_opt.copy()
                
        except Exception as e:
            pass
    
    # Phase 2: Delaunay-based contact graph refinement
    if best_x is not None and time.time() - _START_TIME < deadline - 10.0:
        # Try to refine using Delaunay contact graph
        r_target = best_score * 1.001  # Slightly above current best
        time_for_delaunay = min(20.0, deadline - (time.time() - _START_TIME) - 5.0)
        
        if time_for_delaunay > 5.0:
            x_refined = _delaunay_search(best_x, r_target, num_iterations=30, time_limit=time_for_delaunay)
            if x_refined is not None:
                score_refined = _compute_radius(x_refined)
                if score_refined > best_score:
                    best_score = score_refined
                    best_x = x_refined.copy()
    
    # Phase 3: Final polish with high beta
    if best_x is not None and time.time() - _START_TIME < deadline - 5.0:
        polish_beta = 300.0
        try:
            res = minimize(
                _softmin_obj_and_grad,
                best_x,
                args=(polish_beta,),
                method='L-BFGS-B',
                jac=True,
                bounds=[(-0.5, 0.5)] * (2 * N),
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            x_polished = res.x.reshape(N, 2)
            score_polished = _compute_radius(x_polished)
            
            if score_polished > best_score:
                best_score = score_polished
                best_x = x_polished.copy()
        except Exception as e:
            pass
    
    if best_x is None:
        # Fallback to grid
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        if len(grid) < N:
            best_x = np.random.uniform(-0.4, 0.4, (N, 2))
        else:
            best_x = grid[:N]
            
    return best_x
