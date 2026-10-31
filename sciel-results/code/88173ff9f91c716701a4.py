"""Circle packing in unit square - 26 circles maximizing sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize

N = 26

# EVOLVE-BLOCK-START

def _softplus(x, beta=50.0):
    """Numerically stable softplus."""
    x = np.asarray(x, dtype=float)
    # Avoid overflow in exp
    return np.where(x > 20.0/beta, x, np.log1p(np.exp(beta * x)) / beta)

def _soft_max(v, beta=50.0):
    """Smooth approximation of max(0, v) using softplus."""
    return _softplus(v, beta)

def _objective(x, penalty_weight=500.0, beta=30.0):
    """
    Objective: minimize -sum(radii) + penalty * smooth_violations.
    x = [x1, y1, ..., xN, yN, r1, ..., rN]
    """
    pos = x[:2*N].reshape(N, 2)
    radii = np.maximum(x[2*N:], 1e-8)
    
    # Boundary violations
    # We want x - r >= 0 => r - x <= 0
    # We want x + r <= 1 => x + r - 1 <= 0
    # Penalty on positive violations
    bv_left = radii - pos[:, 0]
    bv_right = pos[:, 0] + radii - 1.0
    bv_bottom = radii - pos[:, 1]
    bv_top = pos[:, 1] + radii - 1.0
    
    # Use soft max to smooth the kink at 0
    boundary_pen = np.sum(_soft_max(bv_left, beta)) + np.sum(_soft_max(bv_right, beta)) + \
                   np.sum(_soft_max(bv_bottom, beta)) + np.sum(_soft_max(bv_top, beta))
    
    # Overlap violations
    # dist(i, j) >= r_i + r_j
    # Equivalent to dist_sq >= (r_i + r_j)^2
    # Violation if (r_i + r_j)^2 - dist_sq > 0
    dx = pos[:, 0, np.newaxis] - pos[np.newaxis, :, 0]
    dy = pos[:, 1, np.newaxis] - pos[np.newaxis, :, 1]
    dist_sq = dx * dx + dy * dy
    min_dist = radii[:, np.newaxis] + radii[np.newaxis, :]
    min_dist_sq = min_dist * min_dist
    
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    
    viol = min_dist_sq - dist_sq
    overlap_pen = np.sum(_soft_max(viol[mask], beta))
    
    return -np.sum(radii) + penalty_weight * (boundary_pen + overlap_pen)

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
    
    # Check if dist < min_dist - tol
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
    radii[0] = 0.25 # Start with a reasonable size, optimizer will adjust
    
    # For subsequent circles, we search for the best location
    # To make it faster and more effective, we use a random search for the position
    # that maximizes the minimum distance to existing circles and walls.
    
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

def construct():
    start_time = time.time()
    deadline = 145.0 # Safety margin
    
    best_params = None
    best_score = 0.0
    
    # Try multiple seeds for greedy initialization
    n_seeds = 10
    
    for seed in range(n_seeds):
        if time.time() > deadline - 10.0:
            break
            
        # 1. Initialization
        init_params = _greedy_init(seed)
        
        # Extract pos and radii for potential local expansion before optimization
        pos = init_params[:2*N].reshape(N, 2)
        radii = init_params[2*N:]
        
        # 2. Pre-optimization expansion
        pos_exp, radii_exp = _radius_expansion(pos, radii)
        init_params_exp = np.zeros(3*N)
        init_params_exp[0:2*N:2] = pos_exp[:, 0]
        init_params_exp[1:2*N:2] = pos_exp[:, 1]
        init_params_exp[2*N:] = radii_exp
        
        # Use the expanded version as start if it's valid or better
        start_params = init_params_exp
        
        # 3. Optimization with L-BFGS-B and soft penalties
        try:
            result = minimize(_objective, start_params, method='L-BFGS-B',
                             options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 5000})
        except Exception as e:
            continue
            
        if time.time() > deadline - 5.0:
            break
            
        res_params = result.x
        res_pos = res_params[:2*N].reshape(N, 2)
        res_radii = res_params[2*N:]
        
        # 4. Post-optimization expansion
        pos_post, radii_post = _radius_expansion(res_pos, res_radii)
        
        # 5. Check validity and score
        # We need to ensure the result is valid.
        # If not strictly valid due to numerical issues, we might shrink slightly.
        # For scoring, we assume the evaluator checks validity.
        # Let's check if it's valid.
        if _is_valid(pos_post, radii_post):
            score = np.sum(radii_post)
            if score > best_score:
                best_score = score
                best_params = np.zeros(3*N)
                best_params[0:2*N:2] = pos_post[:, 0]
                best_params[1:2*N:2] = pos_post[:, 1]
                best_params[2*N:] = radii_post
        else:
            # Try to fix invalidity by shrinking radii slightly
            # This is a heuristic repair
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

    # 6. Final Polish with SLSQP if time permits
    if best_params is not None and time.time() < deadline - 2.0:
        # SLSQP with hard constraints is very effective for polishing
        # Define constraints
        def bounds():
            bounds_list = []
            for i in range(N):
                # x_i - r_i >= 0
                # x_i + r_i <= 1
                # y_i - r_i >= 0
                # y_i + r_i <= 1
                # r_i >= 0
                # We can express these as inequalities g(x) >= 0
                pass
        
        # SLSQP constraints are tricky to vectorize in the definition easily for many circles
        # Given the time, L-BFGS-B with soft penalty + expansion is often sufficient and faster.
        # However, let's try a quick SLSQP polish on the best found so far.
        
        # Constraints for SLSQP:
        # 1. Boundary: x_i - r_i >= 0, etc.
        # 2. Overlap: dist_ij - (r_i + r_j) >= 0
        
        # This is too many constraints for SLSQP to be efficient in the remaining time
        # unless we are very careful. The soft penalty approach already handles this well.
        # Let's stick to the robust L-BFGS-B + Expansion pipeline.
        pass

    if best_params is None:
        # Fallback to a simple valid configuration
        # Grid of 5x5 = 25 circles, plus 1 in center? No, 26.
        # Just use a very small grid to ensure validity.
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
