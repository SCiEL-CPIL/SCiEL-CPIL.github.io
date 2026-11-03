import numpy as np
from scipy.optimize import minimize
import time

N = 21

def _validate_solution(pos_r, w, h, n_circles, tol=1e-6):
    """
    Check if circles are valid: disjoint interiors, within bounds.
    pos_r: (n, 3) array of [x, y, r]
    """
    if pos_r.shape[0] != n_circles:
        return False
    
    xs = pos_r[:, 0]
    ys = pos_r[:, 1]
    rs = pos_r[:, 2]
    
    # Check radii are positive
    if np.any(rs < 1e-9):
        return False
    
    # Check bounds
    if np.any(xs - rs < -tol) or np.any(xs + rs > w + tol):
        return False
    if np.any(ys - rs < -tol) or np.any(ys + rs > h + tol):
        return False
        
    # Check non-overlap
    for i in range(n_circles):
        for j in range(i+1, n_circles):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist = np.sqrt(dx*dx + dy*dy)
            if dist < (rs[i] + rs[j]) - tol:
                return False
    return True

def _hex_init(w, h, n_circles):
    """Generate hexagonal packing initialization."""
    circles = np.zeros((n_circles, 3))
    # Estimate radius for hex packing
    # Area per circle ~ (2*r)^2 * sin(60)/2 for hex
    # For n circles in area w*h, r ~ sqrt(w*h/(n * 2*sqrt(3)))
    r_est = np.sqrt(w * h / (n_circles * 2 * np.sqrt(3)))
    
    # Place in hex pattern
    dx = 2 * r_est
    dy = r_est * np.sqrt(3)
    
    count = 0
    row = 0
    while count < n_circles:
        offset = (dx / 2) if (row % 2) else 0
        col = 0
        while count < n_circles:
            x = offset + dx * col + r_est
            y = dy * row + r_est
            if x + r_est <= w + 1e-6 and y + r_est <= h + 1e-6 and x - r_est >= -1e-6 and y - r_est >= -1e-6:
                circles[count] = [x, y, r_est]
                count += 1
            col += 1
        row += 1
    
    # Fill remaining if needed
    while count < n_circles:
        circles[count] = [w/2, h/2, r_est * 0.5]
        count += 1
    
    return circles

def _random_init(w, h, n_circles, seed=42):
    """Generate random initialization with no overlaps."""
    rng = np.random.RandomState(seed)
    circles = np.zeros((n_circles, 3))
    
    # Start with small radii
    r_min = 0.02
    r_max = min(w, h) / 4
    
    for i in range(n_circles):
        placed = False
        for attempt in range(1000):
            r = rng.uniform(r_min, r_max)
            x = rng.uniform(r, w - r)
            y = rng.uniform(r, h - r)
            
            # Check against existing circles
            ok = True
            for j in range(i):
                dx = x - circles[j, 0]
                dy = y - circles[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < r + circles[j, 2] + 0.01:
                    ok = False
                    break
            
            if ok:
                circles[i] = [x, y, r]
                placed = True
                break
        
        if not placed:
            # Place with minimum radius
            circles[i] = [w/2 + rng.uniform(-0.1, 0.1), h/2 + rng.uniform(-0.1, 0.1), r_min]
    
    return circles

def _optimize_slsqp(pos_r, w, h, time_limit):
    """
    Direct SLSQP optimization to maximize sum of radii.
    Variables: [x1, y1, r1, x2, y2, r2, ..., xn, yn, rn]
    """
    start_time = time.time()
    n = N
    x0 = pos_r.flatten().copy()
    
    # Bounds
    max_r = min(w, h) / 2.0
    bounds = []
    for i in range(n):
        bounds.append((0, w))
        bounds.append((0, h))
        bounds.append((1e-6, max_r))
    
    def objective(x):
        """Minimize negative sum of radii."""
        rs = x[2::3]
        return -np.sum(rs)
    
    def grad_objective(x):
        g = np.zeros_like(x)
        g[2::3] = -1.0
        return g
    
    def constraints(x):
        """List of constraint functions: each returns >= 0."""
        cons = []
        xs = x[0::3]
        ys = x[1::3]
        rs = x[2::3]
        
        # Wall constraints
        for i in range(n):
            # x - r >= 0
            cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[3*i] - x[3*i+2], 'jac': lambda x, i=i: _wall_jac(x, i, 'x_left')})
            # w - x - r >= 0
            cons.append({'type': 'ineq', 'fun': lambda x, i=i: w - x[3*i] - x[3*i+2], 'jac': lambda x, i=i: _wall_jac(x, i, 'x_right')})
            # y - r >= 0
            cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[3*i+1] - x[3*i+2], 'jac': lambda x, i=i: _wall_jac(x, i, 'y_left')})
            # h - y - r >= 0
            cons.append({'type': 'ineq', 'fun': lambda x, i=i: h - x[3*i+1] - x[3*i+2], 'jac': lambda x, i=i: _wall_jac(x, i, 'y_right')})
        
        # Pairwise constraints
        for i in range(n):
            for j in range(i+1, n):
                cons.append({'type': 'ineq', 'fun': lambda x, i=i, j=j: _pair_dist(x, i, j) - x[3*i+2] - x[3*j+2], 
                             'jac': lambda x, i=i, j=j: _pair_jac(x, i, j)})
        
        return cons
    
    def _wall_jac(x, i, wall):
        g = np.zeros_like(x)
        if wall == 'x_left':
            g[3*i] = 1.0
            g[3*i+2] = -1.0
        elif wall == 'x_right':
            g[3*i] = -1.0
            g[3*i+2] = -1.0
        elif wall == 'y_left':
            g[3*i+1] = 1.0
            g[3*i+2] = -1.0
        elif wall == 'y_right':
            g[3*i+1] = -1.0
            g[3*i+2] = -1.0
        return g
    
    def _pair_dist(x, i, j):
        dx = x[3*i] - x[3*j]
        dy = x[3*i+1] - x[3*j+1]
        return np.sqrt(dx*dx + dy*dy)
    
    def _pair_jac(x, i, j):
        g = np.zeros_like(x)
        dx = x[3*i] - x[3*j]
        dy = x[3*i+1] - x[3*j+1]
        dist = np.sqrt(dx*dx + dy*dy)
        if dist < 1e-10:
            dist = 1e-10
            dx = 1e-10
        nx = dx / dist
        ny = dy / dist
        g[3*i] = nx
        g[3*i+1] = ny
        g[3*j] = -nx
        g[3*j+1] = -ny
        return g
    
    def callback(xk):
        return time.time() - start_time > time_limit
    
    try:
        res = minimize(
            objective,
            x0,
            jac=grad_objective,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 100, 'ftol': 1e-10, 'maxfun': 200, 'callback': callback}
        )
        
        result = res.x.reshape(-1, 3)
        
        # Validate and fix small violations
        if _validate_solution(result, w, h, N):
            return result
        
        # Try to fix by shrinking radii slightly
        for i in range(N):
            x, y, r = result[i]
            min_d = min(x, w-x, y, h-y)
            result[i, 2] = min(r, max(1e-6, min_d - 1e-5))
        
        if _validate_solution(result, w, h, N):
            return result
        
        # Return original if failed
        return pos_r
        
    except Exception:
        return pos_r

def _local_search(pos_r, w, h, time_limit, n_restarts=5):
    """
    Perform local search with random perturbations to escape local minima.
    """
    start_time = time.time()
    best = pos_r.copy()
    best_score = np.sum(pos_r[:, 2])
    
    for restart in range(n_restarts):
        if time.time() - start_time > time_limit:
            break
        
        # Perturb positions and radii
        perturbed = best.copy()
        scale = 0.1 * min(w, h)
        perturbed[:, 0] += np.random.uniform(-scale, scale, N)
        perturbed[:, 1] += np.random.uniform(-scale, scale, N)
        perturbed[:, 2] *= np.random.uniform(0.7, 1.0, N)
        
        # Clamp to bounds
        for i in range(N):
            perturbed[i, 0] = np.clip(perturbed[i, 0], 0, w)
            perturbed[i, 1] = np.clip(perturbed[i, 1], 0, h)
            perturbed[i, 2] = np.clip(perturbed[i, 2], 1e-6, min(w, h)/2)
        
        # Optimize from perturbed state
        remaining = time_limit - (time.time() - start_time)
        if remaining < 1:
            break
        
        optimized = _optimize_slsqp(perturbed, w, h, remaining * 0.6)
        
        if _validate_solution(optimized, w, h, N):
            score = np.sum(optimized[:, 2])
            if score > best_score:
                best_score = score
                best = optimized
    
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin before 150s kill
    
    best_result = None
    best_score = 0.0
    
    # Focus on aspect ratios near square and golden ratio
    aspect_ratios = [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.618]
    
    candidates = []
    
    for ar in aspect_ratios:
        if time.time() > deadline - 40:
            break
            
        w = 2.0 * ar / (1 + ar)
        h = 2.0 / (1 + ar)
        
        # Multiple initialization strategies
        inits = []
        
        # Hexagonal
        hex_circles = _hex_init(w, h, N)
        inits.append(('hex', hex_circles))
        
        # Random with different seeds
        for seed in [42, 123, 456]:
            rand_circles = _random_init(w, h, N, seed=seed)
            inits.append(('rand', rand_circles))
        
        for init_name, init_circles in inits:
            if time.time() > deadline - 35:
                break
            
            # Quick SLSQP optimization
            opt_time = min(8.0, deadline - 35 - time.time())
            if opt_time < 2:
                break
            
            optimized = _optimize_slsqp(init_circles, w, h, opt_time)
            
            if _validate_solution(optimized, w, h, N):
                score = np.sum(optimized[:, 2])
                if score > best_score:
                    best_score = score
                    best_result = optimized
                candidates.append((score, optimized, w, h))
                
            if time.time() > deadline - 35:
                break
    
    # Local search on top candidates
    if candidates:
        candidates.sort(key=lambda x: x[0], reverse=True)
        
        for score, pos_r, w, h in candidates[:4]:
            if time.time() > deadline - 10:
                break
            
            ls_time = min(20.0, deadline - 10 - time.time())
            if ls_time < 2:
                break
            
            improved = _local_search(pos_r, w, h, ls_time, n_restarts=3)
            
            if _validate_solution(improved, w, h, N):
                new_score = np.sum(improved[:, 2])
                if new_score > best_score:
                    best_score = new_score
                    best_result = improved

    # Fallback
    if best_result is None:
        w, h = 1.0, 1.0
        best_result = _hex_init(w, h, N)
        for i in range(N):
            x, y, r = best_result[i]
            min_d = min(x, w-x, y, h-y)
            best_result[i, 2] = min(r, max(1e-6, min_d - 1e-4))
        best_result = _optimize_slsqp(best_result, w, h, 10.0)
        if not _validate_solution(best_result, w, h, N):
            # Emergency fix
            for i in range(N):
                x, y, r = best_result[i]
                min_d = min(x, w-x, y, h-y)
                best_result[i, 2] = min(r, max(1e-6, min_d - 1e-3))

    return best_result
