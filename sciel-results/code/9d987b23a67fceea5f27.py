import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import Voronoi, Delaunay

N = 26
EPS = 1e-8
DEADLINE = 145.0  # safety margin before 150s kill


def _validate_and_score(params):
    """
    Check if params (N, 3) is valid and return sum of radii.
    Returns -inf if invalid.
    """
    p = params
    if p.shape != (N, 3):
        return -np.inf
    
    x, y, r = p[:, 0], p[:, 1], p[:, 2]
    
    # Check bounds
    if np.any(r < 0) or np.any(r > 0.5):
        return -np.inf
    if np.any(x < 0) or np.any(x > 1) or np.any(y < 0) or np.any(y > 1):
        return -np.inf
        
    # Check wall constraints
    if np.any(x - r < -EPS) or np.any(1 - x - r < -EPS):
        return -np.inf
    if np.any(y - r < -EPS) or np.any(1 - y - r < -EPS):
        return -np.inf
        
    # Check overlap constraints
    for i in range(N):
        for j in range(i + 1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist_sq = dx*dx + dy*dy
            sum_r = r[i] + r[j]
            if dist_sq < (sum_r - EPS)**2:
                return -np.inf
                
    return np.sum(r)

def _hex_initial(seed):
    """Generate a hexagonal lattice initialization."""
    rng = np.random.RandomState(seed)
    # Scale hex lattice to fit in [0,1]^2
    # 5x5 grid of hex points is 25 points, close to 26.
    # Let's use a denser grid and pick best 26.
    points = []
    for i in range(8):
        for j in range(8):
            x = i * 0.12
            y = j * 0.12 * 1.732 + (0.06 * 1.732 if i % 2 else 0)
            points.append((x, y))
    points = np.array(points)
    
    # Filter points inside [0.05, 0.95] to avoid boundary issues initially
    mask = (points[:, 0] > 0.05) & (points[:, 0] < 0.95) & (points[:, 1] > 0.05) & (points[:, 1] < 0.95)
    points = points[mask]
    
    # If we have more than N, select N that are most spread out or just random
    if len(points) > N:
        # Select N random points from valid ones
        indices = rng.choice(len(points), N, replace=False)
        points = points[indices]
    elif len(points) < N:
        # Pad with random points
        while len(points) < N:
            points = np.vstack([points, rng.uniform(0.1, 0.9, 2)])
            
    # Estimate radii: half min distance to neighbor and wall
    radii = np.zeros(N)
    for i in range(N):
        dists = np.linalg.norm(points - points[i], axis=1)
        dists[i] = np.inf
        min_dist = np.min(dists)
        wall_dist = np.min([points[i][0], 1-points[i][0], points[i][1], 1-points[i][1]])
        radii[i] = 0.5 * np.min([min_dist, wall_dist]) * 0.9
        
    params = np.column_stack((points[:, 0], points[:, 1], radii))
    # Add small jitter
    params += rng.uniform(-0.001, 0.001, params.shape)
    return params

def _random_initial(seed):
    """Generate random initialization."""
    rng = np.random.RandomState(seed)
    points = rng.uniform(0.1, 0.9, (N, 2))
    radii = np.zeros(N)
    for i in range(N):
        dists = np.linalg.norm(points - points[i], axis=1)
        dists[i] = np.inf
        min_dist = np.min(dists)
        wall_dist = np.min([points[i][0], 1-points[i][0], points[i][1], 1-points[i][1]])
        radii[i] = 0.5 * np.min([min_dist, wall_dist]) * 0.9
    params = np.column_stack((points[:, 0], points[:, 1], radii))
    params += rng.uniform(-0.001, 0.001, params.shape)
    return params

def _voronoi_initial(seed):
    """Generate Voronoi-based initialization."""
    rng = np.random.RandomState(seed)
    points = rng.uniform(0.1, 0.9, (N, 2))
    radii = np.zeros(N)
    for i in range(N):
        dists = np.linalg.norm(points - points[i], axis=1)
        dists[i] = np.inf
        min_dist = np.min(dists)
        wall_dist = np.min([points[i][0], 1-points[i][0], points[i][1], 1-points[i][1]])
        radii[i] = 0.5 * np.min([min_dist, wall_dist]) * 0.9
    params = np.column_stack((points[:, 0], points[:, 1], radii))
    params += rng.uniform(-0.001, 0.001, params.shape)
    return params

def _slsqp_optimize(p0, time_limit=10.0):
    """
    Optimize using SLSQP with hard constraints.
    """
    start_time = time.time()
    
    def objective(params):
        return -np.sum(params[2::3])
        
    def objective_grad(params):
        g = np.zeros_like(params)
        g[2::3] = -1.0
        return g
        
    # Constraints:
    # 1. x - r >= 0
    # 2. 1 - x - r >= 0
    # 3. y - r >= 0
    # 4. 1 - y - r >= 0
    # 5. dist(i,j) - (r_i + r_j) >= 0
    
    constraints = []
    
    # Wall constraints
    for i in range(N):
        # x - r >= 0
        def c_x_left(params, i=i):
            return params[3*i] - params[3*i+2]
        def c_x_left_grad(params, i=i):
            g = np.zeros_like(params)
            g[3*i] = 1.0
            g[3*i+2] = -1.0
            return g
        constraints.append({'type': 'ineq', 'fun': c_x_left, 'jac': c_x_left_grad})
        
        # 1 - x - r >= 0
        def c_x_right(params, i=i):
            return 1.0 - params[3*i] - params[3*i+2]
        def c_x_right_grad(params, i=i):
            g = np.zeros_like(params)
            g[3*i] = -1.0
            g[3*i+2] = -1.0
            return g
        constraints.append({'type': 'ineq', 'fun': c_x_right, 'jac': c_x_right_grad})
        
        # y - r >= 0
        def c_y_bottom(params, i=i):
            return params[3*i+1] - params[3*i+2]
        def c_y_bottom_grad(params, i=i):
            g = np.zeros_like(params)
            g[3*i+1] = 1.0
            g[3*i+2] = -1.0
            return g
        constraints.append({'type': 'ineq', 'fun': c_y_bottom, 'jac': c_y_bottom_grad})
        
        # 1 - y - r >= 0
        def c_y_top(params, i=i):
            return 1.0 - params[3*i+1] - params[3*i+2]
        def c_y_top_grad(params, i=i):
            g = np.zeros_like(params)
            g[3*i+1] = -1.0
            g[3*i+2] = -1.0
            return g
        constraints.append({'type': 'ineq', 'fun': c_y_top, 'jac': c_y_top_grad})
        
    # Pairwise constraints
    for i in range(N):
        for j in range(i + 1, N):
            def c_pair(params, i=i, j=j):
                dx = params[3*i] - params[3*j]
                dy = params[3*i+1] - params[3*j+1]
                dist = np.sqrt(dx*dx + dy*dy)
                return dist - (params[3*i+2] + params[3*j+2])
            def c_pair_grad(params, i=i, j=j):
                dx = params[3*i] - params[3*j]
                dy = params[3*i+1] - params[3*j+1]
                dist_sq = dx*dx + dy*dy
                if dist_sq < 1e-10:
                    dist_sq = 1e-10
                dist = np.sqrt(dist_sq)
                ux = dx / dist
                uy = dy / dist
                g = np.zeros_like(params)
                g[3*i] = ux
                g[3*j] = -ux
                g[3*i+1] = uy
                g[3*j+1] = -uy
                g[3*i+2] = -1.0
                g[3*j+2] = -1.0
                return g
            constraints.append({'type': 'ineq', 'fun': c_pair, 'jac': c_pair_grad})
            
    bounds = []
    for i in range(N):
        bounds.append((0.0, 1.0)) # x
        bounds.append((0.0, 1.0)) # y
        bounds.append((1e-6, 0.5)) # r
        
    p0_flat = p0.flatten()
    
    try:
        res = minimize(
            objective,
            p0_flat,
            method='SLSQP',
            jac=objective_grad,
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 200, 'ftol': 1e-8}
        )
        
        if time.time() - start_time > time_limit:
            return p0, _validate_and_score(p0)
            
        if res.success:
            p_new = res.x.reshape(N, 3)
            score = _validate_and_score(p_new)
            if score > _validate_and_score(p0):
                return p_new, score
        return p0, _validate_and_score(p0)
    except Exception:
        return p0, _validate_and_score(p0)

def _scale_and_refine(p, time_limit=10.0):
    """
    Try to scale up radii and re-optimize.
    """
    current_score = _validate_and_score(p)
    if current_score < 0:
        return p, current_score
        
    best_p = p.copy()
    best_score = current_score
    
    # Try scaling factors
    for scale in [1.01, 1.02, 1.05, 1.1]:
        if time.time() > time.time() + time_limit: # Just a guard, actual check below
            break
            
        p_scaled = best_p.copy()
        p_scaled[:, 2] *= scale
        
        # This might be invalid, so we rely on SLSQP to fix it? 
        # SLSQP starts from a point. If the point is invalid, it might fail or converge to a different local min.
        # It's safer to use the optimizer to push it back into feasibility while maximizing radius.
        # However, SLSQP requires a feasible starting point for best performance.
        # Let's just run SLSQP on the scaled radii. It will try to move centers to accommodate larger radii.
        
        p_opt, s_opt = _slsqp_optimize(p_scaled, time_limit=max(1.0, time_limit - (time.time() - (time.time() - time_limit))))
        
        if s_opt > best_score:
            best_score = s_opt
            best_p = p_opt
        else:
            break # If scaling didn't help, stop
            
    return best_p, best_score

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    best_params = None
    best_score = -np.inf
    
    initializers = [
        _hex_initial,
        _random_initial,
        _voronoi_initial
    ]
    
    seeds = [42, 123, 7, 999, 555, 2024, 31415, 271828, 1234, 5678, 11, 22, 33, 44, 55]
    
    for seed in seeds:
        if time.time() > deadline - 15:
            break
            
        for init_fn in initializers:
            if time.time() > deadline - 15:
                break
                
            p0 = init_fn(seed)
            
            # Initial optimization
            time_remaining = deadline - time.time() - 10
            if time_remaining < 5:
                continue
                
            p_opt, s_opt = _slsqp_optimize(p0, time_limit=min(8.0, time_remaining))
            
            if s_opt > best_score:
                best_score = s_opt
                best_params = p_opt.copy()
                
            # If we have a good score, try scaling
            if best_score > 0 and time.time() < deadline - 20:
                p_scaled, s_scaled = _scale_and_refine(best_params, time_limit=min(10.0, deadline - time.time() - 5))
                if s_scaled > best_score:
                    best_score = s_scaled
                    best_params = p_scaled.copy()

    if best_params is None:
        # Fallback
        p0 = _hex_initial(42)
        best_params, best_score = _slsqp_optimize(p0, time_limit=5.0)
        
    return best_params
