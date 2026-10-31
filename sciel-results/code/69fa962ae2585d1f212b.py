"""Circle packing in unit square - 26 circles maximizing sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize

N = 26

# EVOLVE-BLOCK-START

def _is_valid(pos, radii, tol=1e-7):
    """Strict validity check."""
    n = len(radii)
    if np.any(radii <= 0):
        return False
    if np.any(pos[:, 0] - radii < -tol) or np.any(pos[:, 0] + radii > 1.0 + tol):
        return False
    if np.any(pos[:, 1] - radii < -tol) or np.any(pos[:, 1] + radii > 1.0 + tol):
        return False
        
    dx = pos[:, 0, np.newaxis] - pos[np.newaxis, :, 0]
    dy = pos[:, 1, np.newaxis] - pos[np.newaxis, :, 1]
    dist_sq = dx * dx + dy * dy
    min_dist = radii[:, np.newaxis] + radii[np.newaxis, :]
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    
    if np.any(dist_sq[mask] < (min_dist[mask] - tol)**2):
        return False
    return True

def _greedy_init(seed=0):
    """
    Greedy best-fit initialization.
    Places circles one by one, trying to maximize the radius of the next circle
    given the previous ones.
    """
    rng = np.random.RandomState(seed)
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    
    # Place first circle in center
    pos[0] = [0.5, 0.5]
    radii[0] = 0.25
    
    for i in range(1, N):
        best_x, best_y, best_r = 0.5, 0.5, 0.0
        # Sample many candidate positions
        for _ in range(2000):
            x = rng.uniform(0.05, 0.95)
            y = rng.uniform(0.05, 0.95)
            
            # Max radius allowed by walls
            max_r_wall = min(x, 1.0 - x, y, 1.0 - y)
            
            # Max radius allowed by existing circles
            max_r_circ = max_r_wall
            for j in range(i):
                dx = x - pos[j, 0]
                dy = y - pos[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                possible_r = dist - radii[j]
                if possible_r < max_r_circ:
                    max_r_circ = possible_r
            
            if max_r_circ > best_r and max_r_circ > 0.01:
                best_r = max_r_circ
                best_x = x
                best_y = y
                
        pos[i] = [best_x, best_y]
        radii[i] = best_r
        
    # Convert to parameter vector
    params = np.zeros(3*N)
    params[0:2*N:2] = pos[:, 0]
    params[1:2*N:2] = pos[:, 1]
    params[2*N:] = radii
    return params

def _radius_expansion(pos, radii, max_iter=50):
    """
    Try to expand radii of circles that have space.
    This is a local improvement step.
    """
    pos = pos.copy()
    radii = radii.copy()
    
    for _ in range(max_iter):
        improved = False
        for i in range(N):
            # Check if we can expand circle i
            # Limit by walls
            max_r_wall = min(pos[i, 0], 1.0 - pos[i, 0], pos[i, 1], 1.0 - pos[i, 1])
            current_r = radii[i]
            
            if current_r >= max_r_wall - 1e-6:
                continue
                
            # Limit by other circles
            max_r_circ = max_r_wall
            for j in range(N):
                if i == j:
                    continue
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                space = dist - radii[j]
                if space < max_r_circ:
                    max_r_circ = space
            
            if max_r_circ > current_r + 1e-5:
                # Expand by a fraction of the gap
                expansion = (max_r_circ - current_r) * 0.5
                radii[i] += expansion
                improved = True
        
        if not improved:
            break
            
    return pos, radii

def _slsqp_polish(pos, radii, time_limit=10.0):
    """
    Polish a valid configuration using SLSQP with hard constraints.
    This ensures feasibility while maximizing the sum of radii.
    """
    start_time = time.time()
    
    # Flatten
    x0 = np.zeros(3*N)
    x0[0:2*N:2] = pos[:, 0]
    x0[1:2*N:2] = pos[:, 1]
    x0[2*N:] = radii
    
    # Objective: maximize sum(r) -> minimize -sum(r)
    def objective(x):
        return -np.sum(x[2*N:])
    
    # Gradient of objective
    def gradient(x):
        g = np.zeros_like(x)
        g[2*N:] = -1.0
        return g
    
    # Bounds: x, y in [0, 1], r in [0, 0.5]
    bounds = []
    for i in range(N):
        bounds.append((0.0, 1.0))  # x_i
        bounds.append((0.0, 1.0))  # y_i
    for i in range(N):
        bounds.append((1e-6, 0.5))  # r_i
        
    # Constraints:
    # 1. Boundary: x_i - r_i >= 0, etc.
    # 2. Overlap: dist_ij - (r_i + r_j) >= 0
    
    constraints = []
    
    # Boundary constraints
    for i in range(N):
        xi = 2*i
        yi = 2*i + 1
        ri = 2*N + i
        
        # x_i - r_i >= 0
        constraints.append({
            'type': 'ineq',
            'fun': lambda x, xi=xi, ri=ri: x[xi] - x[ri],
            'jac': lambda x, xi=xi, ri=ri: np.concatenate([np.eye(1)[0], np.zeros(xi), np.eye(1)[0], -np.eye(1)[0], np.zeros(2*N + ri - xi - 1)])[:3*N] if False else (lambda x, xi=xi, ri=ri: np.concatenate([[1.0] + [0.0]*xi + [-1.0] + [0.0]*(2*N + ri - xi - 1)])[:3*N])(x)
        })
        # x_i + r_i <= 1 -> 1 - x_i - r_i >= 0
        constraints.append({
            'type': 'ineq',
            'fun': lambda x, xi=xi, ri=ri: 1.0 - x[xi] - x[ri],
            'jac': lambda x, xi=xi, ri=ri: np.concatenate([[-1.0] + [0.0]*xi + [-1.0] + [0.0]*(2*N + ri - xi - 1)])[:3*N] if False else (lambda x, xi=xi, ri=ri: np.concatenate([[-1.0] + [0.0]*xi + [-1.0] + [0.0]*(2*N + ri - xi - 1)])[:3*N])(x)
        })
        
        # y_i - r_i >= 0
        constraints.append({
            'type': 'ineq',
            'fun': lambda x, yi=yi, ri=ri: x[yi] - x[ri],
            'jac': lambda x, yi=yi, ri=ri: (lambda x, yi=yi, ri=ri: np.concatenate([[0.0]*yi + [1.0] + [0.0]*(2*N + ri - yi - 1)])[:3*N])(x)
        })
        # y_i + r_i <= 1
        constraints.append({
            'type': 'ineq',
            'fun': lambda x, yi=yi, ri=ri: 1.0 - x[yi] - x[ri],
            'jac': lambda x, yi=yi, ri=ri: (lambda x, yi=yi, ri=ri: np.concatenate([[0.0]*yi + [-1.0] + [0.0]*(2*N + ri - yi - 1)])[:3*N])(x)
        })
        
    # Overlap constraints
    # This is O(N^2) constraints, which is heavy for SLSQP.
    # We only add them if N is small enough or we use a subset.
    # For N=26, that's 325 constraints. SLSQP might struggle or be slow.
    # Let's try it but with a tight time limit.
    
    for i in range(N):
        for j in range(i+1, N):
            xi = 2*i
            yi = 2*i + 1
            xj = 2*j
            yj = 2*j + 1
            ri = 2*N + i
            rj = 2*N + j
            
            # dist - (ri + rj) >= 0
            # dist = sqrt((xi-xj)^2 + (yi-yj)^2)
            # Gradient of dist:
            # d(dist)/d(xi) = (xi-xj)/dist
            # d(dist)/d(xj) = (xj-xi)/dist
            # d(dist)/d(ri) = -1
            # d(dist)/d(rj) = -1
            
            def fun(x, xi=xi, yi=yi, xj=xj, yj=yj, ri=ri, rj=rj):
                dx = x[xi] - x[xj]
                dy = x[yi] - x[yj]
                dist = np.sqrt(dx*dx + dy*dy)
                return dist - x[ri] - x[rj]
            
            def jac(x, xi=xi, yi=yi, xj=xj, yj=yj, ri=ri, rj=rj):
                dx = x[xi] - x[xj]
                dy = x[yi] - x[yj]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < 1e-12:
                    # Handle singularity
                    grad = np.zeros(3*N)
                    grad[xi] = 1.0
                    grad[xj] = -1.0
                    grad[ri] = -1.0
                    grad[rj] = -1.0
                    return grad
                
                grad = np.zeros(3*N)
                grad[xi] = dx / dist
                grad[yi] = dy / dist
                grad[xj] = -dx / dist
                grad[yj] = -dy / dist
                grad[ri] = -1.0
                grad[rj] = -1.0
                return grad
            
            constraints.append({
                'type': 'ineq',
                'fun': fun,
                'jac': jac
            })
            
            if time.time() - start_time > time_limit:
                break
        if time.time() - start_time > time_limit:
            break
    
    try:
        result = minimize(objective, x0, method='SLSQP', jac=gradient,
                          bounds=bounds, constraints=constraints,
                          options={'maxiter': 100, 'ftol': 1e-10})
        
        if result.success or result.fun < objective(x0):
            res_pos = result.x[:2*N].reshape(N, 2)
            res_radii = result.x[2*N:]
            if _is_valid(res_pos, res_radii):
                return res_pos, res_radii
    except Exception:
        pass
        
    return pos, radii

def construct():
    start_time = time.time()
    deadline = 145.0 # Safety margin
    
    best_params = None
    best_score = 0.0
    
    # Try multiple seeds for greedy initialization
    n_seeds = 5 # Reduced to allow time for SLSQP
    
    for seed in range(n_seeds):
        if time.time() > deadline - 15.0:
            break
            
        # 1. Initialization
        init_params = _greedy_init(seed)
        
        # Extract pos and radii for potential local expansion before optimization
        pos = init_params[:2*N].reshape(N, 2)
        radii = init_params[2*N:]
        
        # 2. Pre-optimization expansion
        pos_exp, radii_exp = _radius_expansion(pos, radii)
        
        # 3. Optimization with L-BFGS-B and soft penalties (kept from current for robustness)
        # But we will rely more on SLSQP for the final polish.
        # Let's do a quick L-BFGS-B to get a better starting point for SLSQP.
        
        # Objective for L-BFGS-B (soft penalty)
        def _objective(x, penalty_weight=500.0, beta=30.0):
            pos = x[:2*N].reshape(N, 2)
            radii = np.maximum(x[2*N:], 1e-8)
            
            bv_left = radii - pos[:, 0]
            bv_right = pos[:, 0] + radii - 1.0
            bv_bottom = radii - pos[:, 1]
            bv_top = pos[:, 1] + radii - 1.0
            
            boundary_pen = np.sum(np.maximum(bv_left, 0)) + np.sum(np.maximum(bv_right, 0)) + \
                           np.sum(np.maximum(bv_bottom, 0)) + np.sum(np.maximum(bv_top, 0))
            
            dx = pos[:, 0, np.newaxis] - pos[np.newaxis, :, 0]
            dy = pos[:, 1, np.newaxis] - pos[np.newaxis, :, 1]
            dist_sq = dx * dx + dy * dy
            min_dist = radii[:, np.newaxis] + radii[np.newaxis, :]
            min_dist_sq = min_dist * min_dist
            mask = np.triu(np.ones((N, N), dtype=bool), k=1)
            
            viol = min_dist_sq - dist_sq
            overlap_pen = np.sum(np.maximum(viol[mask], 0))
            
            return -np.sum(radii) + penalty_weight * (boundary_pen + overlap_pen)

        start_params = np.zeros(3*N)
        start_params[0:2*N:2] = pos_exp[:, 0]
        start_params[1:2*N:2] = pos_exp[:, 1]
        start_params[2*N:] = radii_exp
        
        try:
            result = minimize(_objective, start_params, method='L-BFGS-B',
                             options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-6, 'maxfun': 2000})
        except Exception:
            result = None
        
        if result is not None:
            res_params = result.x
            res_pos = res_params[:2*N].reshape(N, 2)
            res_radii = res_params[2*N:]
            
            # Post-optimization expansion
            pos_post, radii_post = _radius_expansion(res_pos, res_radii)
            
            # Check validity
            if _is_valid(pos_post, radii_post):
                score = np.sum(radii_post)
                if score > best_score:
                    best_score = score
                    best_params = np.zeros(3*N)
                    best_params[0:2*N:2] = pos_post[:, 0]
                    best_params[1:2*N:2] = pos_post[:, 1]
                    best_params[2*N:] = radii_post
                    
                    # 4. SLSQP Polish if time permits
                    if time.time() < deadline - 5.0:
                        polished_pos, polished_radii = _slsqp_polish(pos_post, radii_post, time_limit=10.0)
                        if _is_valid(polished_pos, polished_radii):
                            polish_score = np.sum(polished_radii)
                            if polish_score > best_score:
                                best_score = polish_score
                                best_params = np.zeros(3*N)
                                best_params[0:2*N:2] = polished_pos[:, 0]
                                best_params[1:2*N:2] = polished_pos[:, 1]
                                best_params[2*N:] = polished_radii
            else:
                # Repair if invalid
                repair_pos, repair_radii = pos_post.copy(), radii_post.copy()
                for _ in range(100):
                    valid = True
                    for i in range(N):
                        x, y = repair_pos[i]
                        r = repair_radii[i]
                        max_r = min(x, 1.0-x, y, 1.0-y)
                        if r > max_r:
                            repair_radii[i] = max_r * 0.999
                            valid = False
                    for i in range(N):
                        for j in range(i+1, N):
                            dx = repair_pos[i,0] - repair_pos[j,0]
                            dy = repair_pos[i,1] - repair_pos[j,1]
                            dist = np.sqrt(dx*dx + dy*dy)
                            sum_r = repair_radii[i] + repair_radii[j]
                            if dist < sum_r:
                                k = dist / (sum_r + 1e-12) * 0.999
                                repair_radii[i] *= k
                                repair_radii[j] *= k
                                valid = False
                    if valid:
                        break
                    
                    if _is_valid(repair_pos, repair_radii):
                        score = np.sum(repair_radii)
                        if score > best_score:
                            best_score = score
                            best_params = np.zeros(3*N)
                            best_params[0:2*N:2] = repair_pos[:, 0]
                            best_params[1:2*N:2] = repair_pos[:, 1]
                            best_params[2*N:] = repair_radii

    # 5. Final Polish with SLSQP on the best found so far
    if best_params is not None and time.time() < deadline - 2.0:
        best_pos = best_params[0:2*N:2].reshape(N, 2)
        best_radii = best_params[2*N:]
        polished_pos, polished_radii = _slsqp_polish(best_pos, best_radii, time_limit=10.0)
        if _is_valid(polished_pos, polished_radii):
            polish_score = np.sum(polished_radii)
            if polish_score > best_score:
                best_score = polish_score
                best_params = np.zeros(3*N)
                best_params[0:2*N:2] = polished_pos[:, 0]
                best_params[1:2*N:2] = polished_pos[:, 1]
                best_params[2*N:] = polished_radii

    if best_params is None:
        # Fallback
        side = 6
        params = np.zeros(3*N)
        idx = 0
        for r in range(side):
            for c in range(side):
                if idx >= N:
                    break
                x = (c + 0.5) / side
                y = (r + 0.5) / side
                rad = 0.4 / side
                params[3*idx] = x
                params[3*idx+1] = y
                params[3*idx+2] = rad
                idx += 1
        return params.reshape(N, 3)
        
    # Format output
    output = np.zeros((N, 3))
    output[:, 0] = best_params[0:2*N:2]
    output[:, 1] = best_params[1:2*N:2]
    output[:, 2] = best_params[2*N:]
    
    return output

# EVOLVE-BLOCK-END
