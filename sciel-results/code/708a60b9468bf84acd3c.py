<<<<<<< SEARCH
import numpy as np
import time
from scipy.optimize import minimize
import itertools

N = 26

def _penalty_and_objective(params, N):
    """
    Computes the objective function to minimize.
    We want to maximize sum(r), so we minimize -sum(r).
    We add penalties for boundary violations and overlaps.
    params: array of shape (N, 3) with columns [x, y, r]
    """
    x = params[:, 0]
    y = params[:, 1]
    r = params[:, 2]
    
    # Objective: Maximize sum of radii
    obj = -np.sum(r)
    
    # Penalty for boundary violations
    # x >= r, x <= 1-r, y >= r, y <= 1-r
    # Equivalent to: r <= min(x, 1-x, y, 1-y)
    # Violation = max(0, r - min(x, 1-x, y, 1-y))
    min_dist_to_boundary = np.minimum(np.minimum(x, 1 - x), np.minimum(y, 1 - y))
    boundary_violation = np.maximum(0, r - min_dist_to_boundary)
    penalty_boundary = np.sum(boundary_violation)
    
    # Penalty for overlaps
    # Distance d_ij must be >= r_i + r_j
    # Violation = max(0, (r_i + r_j) - d_ij)
    # Using squared penalty for smoothness: max(0, (r_i + r_j) - d_ij)^2
    penalty_overlap = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d = np.sqrt(dx*dx + dy*dy)
            min_dist = r[i] + r[j]
            if d < min_dist:
                penalty_overlap += (min_dist - d)**2
                
    # Weights
    w_bound = 100.0
    w_overlap = 100.0
    
    return obj + w_bound * penalty_boundary + w_overlap * penalty_overlap

def construct():
    start_time = time.time()
    timeout = 110.0  # Safety margin
    
    # Initial configuration: Grid
    side = int(np.ceil(np.sqrt(N)))
    cells = []
    for i in range(side):
        for j in range(side):
            if len(cells) < N:
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                r = 0.5 / side - 1e-4
                cells.append([x, y, r])
    
    best_params = np.array(cells[:N])
    best_score = -np.sum(best_params[:, 2])  # We minimize -sum(r), so score is -obj
    
    # Helper to check validity and get score
    def get_score(params):
        x = params[:, 0]
        y = params[:, 1]
        r = params[:, 2]
        
        # Check boundaries
        if np.any(x < r - 1e-5) or np.any(x > 1 - r + 1e-5) or np.any(y < r - 1e-5) or np.any(y > 1 - r + 1e-5):
            return -np.inf
        # Check overlaps
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx*dx + dy*dy)
                if d < r[i] + r[j] - 1e-5:
                    return -np.inf
        return np.sum(r)

    best_valid_score = get_score(best_params)
    
    rng = np.random.default_rng(42)
    
    while time.time() - start_time < timeout:
        # Create a perturbed initial guess
        params = best_params.copy()
        # Add small random noise to positions and radii
        noise_scale = 0.05
        params[:, 0] += rng.uniform(-noise_scale, noise_scale, N)
        params[:, 1] += rng.uniform(-noise_scale, noise_scale, N)
        params[:, 2] += rng.uniform(-noise_scale, noise_scale, N)
        
        # Clip to reasonable bounds to help optimizer
        params[:, 0] = np.clip(params[:, 0], 0.01, 0.99)
        params[:, 1] = np.clip(params[:, 1], 0.01, 0.99)
        params[:, 2] = np.clip(params[:, 2], 0.01, 0.4)
        
        # Optimize
        try:
            res = minimize(
                _penalty_and_objective,
                params,
                args=(N,),
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-6}
            )
            
            if res.success:
                candidate = res.x.reshape(N, 3)
                # Ensure radii are positive
                candidate[:, 2] = np.maximum(candidate[:, 2], 1e-6)
                
                # Check validity
                score = get_score(candidate)
                if score > best_valid_score:
                    best_valid_score = score
                    best_params = candidate
        except Exception:
            pass
            
        # Check time
        if time.time() - start_time > timeout:
            break
            
    # Final cleanup: ensure strict validity by shrinking slightly if needed
    # The penalty method might leave tiny violations.
    # We can do a final "fix" pass:
    # 1. If any circle is out of bounds, shrink it or move it.
    # 2. If any circles overlap, shrink the smaller one or move them.
    
    # Simple iterative fix for final result
    final_params = best_params.copy()
    for _ in range(10):
        changed = False
        # Fix boundaries
        for i in range(N):
            min_dist = min(final_params[i, 0], 1 - final_params[i, 0], final_params[i, 1], 1 - final_params[i, 1])
            if final_params[i, 2] > min_dist:
                final_params[i, 2] = min_dist
                changed = True
        
        # Fix overlaps
        for i in range(N):
            for j in range(i + 1, N):
                dx = final_params[i, 0] - final_params[j, 0]
                dy = final_params[i, 1] - final_params[j, 1]
                d = np.sqrt(dx*dx + dy*dy)
                min_dist = final_params[i, 2] + final_params[j, 2]
                if d < min_dist:
                    # Shrink the smaller radius to resolve overlap
                    # Or scale both? Shrink smaller is safer for sum if one is much larger?
                    # Actually, to maximize sum, we should probably shrink both proportionally or just the smaller one.
                    # Let's shrink the smaller one to fit.
                    if final_params[i, 2] <= final_params[j, 2]:
                        # Reduce r_i so that r_i + r_j = d
                        new_r_i = d - final_params[j, 2]
                        if new_r_i > 0:
                            final_params[i, 2] = new_r_i
                        else:
                            # If d is too small, we must shrink both or move them.
                            # Moving is complex here. Let's just shrink both to half the distance.
                            final_params[i, 2] = d / 2 * 0.99
                            final_params[j, 2] = d / 2 * 0.99
                        changed = True
                    else:
                        new_r_j = d - final_params[i, 2]
                        if new_r_j > 0:
                            final_params[j, 2] = new_r_j
                        else:
                            final_params[i, 2] = d / 2 * 0.99
                            final_params[j, 2] = d / 2 * 0.99
                        changed = True
        if not changed:
            break
            
    return final_params
=======
import numpy as np
import time

N = 26
DEADLINE = 140.0  # Safety margin from 150s wall-clock limit

def _elapsed(start_time):
    return time.time() - start_time

def _eval_packing(params):
    """
    Evaluate a packing configuration.
    Returns (sum_radii, is_valid).
    """
    x = params[:, 0]
    y = params[:, 1]
    r = params[:, 2]
    n = len(r)
    
    # Check boundaries
    min_dist = np.minimum(np.minimum(x, 1 - x), np.minimum(y, 1 - y))
    if np.any(r > min_dist + 1e-9):
        return 0.0, False
    
    # Check overlaps (vectorized)
    for i in range(n):
        dx = x[i] - x[i+1:]
        dy = y[i] - y[i+1:]
        d = np.sqrt(dx*dx + dy*dy)
        if np.any(d < r[i] + r[i+1:] - 1e-9):
            return 0.0, False
    
    return float(np.sum(r)), True

def _grid_init():
    """Create initial grid configuration."""
    side = int(np.ceil(np.sqrt(N)))
    cells = []
    for i in range(side):
        for j in range(side):
            if len(cells) < N:
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                r = 0.5 / side - 1e-4
                cells.append([x, y, r])
    return np.array(cells[:N])

def _random_init(rng):
    """Create random initial configuration."""
    params = np.zeros((N, 3))
    params[:, 0] = rng.uniform(0.05, 0.95, N)
    params[:, 1] = rng.uniform(0.05, 0.95, N)
    params[:, 2] = rng.uniform(0.02, 0.08, N)
    return params

def _max_radius_for_circle(i, params, grid_xs, grid_ys):
    """
    Find the maximum radius for circle i given its position and other circles.
    Uses a grid of candidate positions to find the best fit.
    """
    x_i = params[i, 0]
    y_i = params[i, 1]
    
    # Max radius from boundaries
    r_bound = min(x_i, 1 - x_i, y_i, 1 - y_i)
    
    # Min radius from other circles
    r_max = r_bound
    for j in range(N):
        if j == i:
            continue
        dx = x_i - params[j, 0]
        dy = y_i - params[j, 1]
        d = np.sqrt(dx*dx + dy*dy)
        r_from_j = d - params[j, 2]
        if r_from_j < r_max:
            r_max = r_from_j
    
    return max(0.0, r_max)

def _best_fit_local_search(params, start_time, max_iter=50):
    """
    Best-fit local search: for each circle, try moving it to nearby grid positions
    and keep the one that gives the largest possible radius.
    """
    current_params = params.copy()
    n = len(current_params)
    
    # Create a grid of candidate positions
    grid_size = 121  # 11x11 grid
    grid_step = 0.08
    grid_xs = np.linspace(0.02, 0.98, grid_size)
    grid_ys = np.linspace(0.02, 0.98, grid_size)
    
    improved = True
    iters = 0
    
    while improved and iters < max_iter:
        improved = False
        iters += 1
        
        if _elapsed(start_time) > DEADLINE - 10:
            break
        
        for i in range(n):
            x_i = current_params[i, 0]
            y_i = current_params[i, 1]
            r_i = current_params[i, 2]
            
            best_new_r = r_i
            best_new_x = x_i
            best_new_y = y_i
            
            # Try nearby grid positions
            # Find closest grid indices
            idx_x = np.argmin(np.abs(grid_xs - x_i))
            idx_y = np.argmin(np.abs(grid_ys - y_i))
            
            # Search in a window around current position
            window = 3
            for gx in range(max(0, idx_x - window), min(grid_size, idx_x + window + 1)):
                for gy in range(max(0, idx_y - window), min(grid_size, idx_y + window + 1)):
                    new_x = grid_xs[gx]
                    new_y = grid_ys[gy]
                    
                    # Calculate max radius at this position
                    r_bound = min(new_x, 1 - new_x, new_y, 1 - new_y)
                    r_max = r_bound
                    
                    for j in range(n):
                        if j == i:
                            continue
                        dx = new_x - current_params[j, 0]
                        dy = new_y - current_params[j, 1]
                        d = np.sqrt(dx*dx + dy*dy)
                        r_from_j = d - current_params[j, 2]
                        if r_from_j < r_max:
                            r_max = r_from_j
                    
                    r_candidate = max(0.0, r_max)
                    
                    if r_candidate > best_new_r + 1e-10:
                        best_new_r = r_candidate
                        best_new_x = new_x
                        best_new_y = new_y
            
            if best_new_r > r_i + 1e-10:
                current_params[i, 0] = best_new_x
                current_params[i, 1] = best_new_y
                current_params[i, 2] = best_new_r
                improved = True
    
    return current_params

def _gradient_polish(params, start_time, max_iter=100):
    """
    Gradient-based polish using smooth objective.
    """
    current_params = params.copy()
    n = len(current_params)
    
    # Smooth objective: -sum(r) + penalty for overlaps
    def smooth_obj(p):
        x = p[:n]
        y = p[n:2*n]
        r = p[2*n:3*n]
        
        # Boundary penalty
        min_dist = np.minimum(np.minimum(x, 1 - x), np.minimum(y, 1 - y))
        bound_pen = np.sum(np.maximum(0, r - min_dist))
        
        # Overlap penalty (smooth)
        overlap_pen = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx*dx + dy*dy + 1e-12)
                min_d = r[i] + r[j]
                if d < min_d:
                    overlap_pen += (min_d - d)**2
        
        return -np.sum(r) + 50.0 * bound_pen + 50.0 * overlap_pen
    
    def smooth_grad(p):
        x = p[:n]
        y = p[n:2*n]
        r = p[2*n:3*n]
        
        grad = np.zeros(3 * n)
        
        # Gradient of -sum(r)
        grad[2*n:3*n] = -1.0
        
        # Gradient of boundary penalty
        min_dist = np.minimum(np.minimum(x, 1 - x), np.minimum(y, 1 - y))
        for i in range(n):
            if r[i] > min_dist[i]:
                # Which constraint is active?
                if min_dist[i] == x[i]:
                    grad[i] += 50.0
                if min_dist[i] == 1 - x[i]:
                    grad[i] -= 50.0
                if min_dist[i] == y[i]:
                    grad[n + i] += 50.0
                if min_dist[i] == 1 - y[i]:
                    grad[n + i] -= 50.0
                grad[2*n + i] += 50.0
        
        # Gradient of overlap penalty
        for i in range(n):
            for j in range(i + 1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx*dx + dy*dy + 1e-12)
                min_d = r[i] + r[j]
                if d < min_d:
                    diff = min_d - d
                    factor = 2 * diff / d
                    grad[i] += 50.0 * factor * dx
                    grad[n + i] += 50.0 * factor * dy
                    grad[2*n + i] += 50.0 * diff
                    grad[j] -= 50.0 * factor * dx
                    grad[n + j] -= 50.0 * factor * dy
                    grad[2*n + j] += 50.0 * diff
        
        return grad
    
    # Simple gradient descent with adaptive step
    p = current_params.flatten()
    step_size = 0.01
    
    for it in range(max_iter):
        if _elapsed(start_time) > DEADLINE - 5:
            break
        
        obj = smooth_obj(p)
        grad = smooth_grad(p)
        
        # Adaptive step size
        p_new = p - step_size * grad
        
        # Clip to reasonable bounds
        p_new[:n] = np.clip(p_new[:n], 0.001, 0.999)
        p_new[n:2*n] = np.clip(p_new[n:2*n], 0.001, 0.999)
        p_new[2*n:3*n] = np.clip(p_new[2*n:3*n], 0.001, 0.45)
        
        obj_new = smooth_obj(p_new)
        
        if obj_new < obj:
            p = p_new
            step_size = min(step_size * 1.05, 0.05)
        else:
            step_size *= 0.9
    
    return p.reshape(n, 3)

def _fix_validity(params):
    """Ensure the packing is strictly valid by adjusting radii."""
    final_params = params.copy()
    
    for _ in range(20):
        changed = False
        
        # Fix boundaries
        for i in range(N):
            min_dist = min(final_params[i, 0], 1 - final_params[i, 0], 
                          final_params[i, 1], 1 - final_params[i, 1])
            if final_params[i, 2] > min_dist - 1e-10:
                final_params[i, 2] = max(0.0, min_dist - 1e-6)
                changed = True
        
        # Fix overlaps
        for i in range(N):
            for j in range(i + 1, N):
                dx = final_params[i, 0] - final_params[j, 0]
                dy = final_params[i, 1] - final_params[j, 1]
                d = np.sqrt(dx*dx + dy*dy)
                min_dist = final_params[i, 2] + final_params[j, 2]
                if d < min_dist - 1e-10:
                    # Shrink both proportionally
                    scale = d / min_dist * 0.999
                    final_params[i, 2] *= scale
                    final_params[j, 2] *= scale
                    changed = True
        
        if not changed:
            break
    
    return final_params

def construct():
    start_time = time.time()
    rng = np.random.default_rng(42)
    
    # Start with grid initialization
    best_params = _grid_init()
    best_score, _ = _eval_packing(best_params)
    
    # Run best-fit local search
    best_params = _best_fit_local_search(best_params, start_time, max_iter=30)
    score, _ = _eval_packing(best_params)
    if score > best_score:
        best_score = score
    
    # Gradient polish
    polished = _gradient_polish(best_params, start_time, max_iter=80)
    polished = _fix_validity(polished)
    score, _ = _eval_packing(polished)
    if score > best_score:
        best_params = polished
        best_score = score
    
    # Try more iterations if time allows
    while _elapsed(start_time) < DEADLINE - 20:
        # Perturb and re-search
        params = best_params.copy()
        noise = 0.03
        params[:, 0] += rng.uniform(-noise, noise, N)
        params[:, 1] += rng.uniform(-noise, noise, N)
        params[:, 2] += rng.uniform(-noise, noise, N)
        params[:, 0] = np.clip(params[:, 0], 0.01, 0.99)
        params[:, 1] = np.clip(params[:, 1], 0.01, 0.99)
        params[:, 2] = np.clip(params[:, 2], 0.01, 0.4)
        
        # Best-fit local search
        params = _best_fit_local_search(params, start_time, max_iter=15)
        
        # Gradient polish
        params = _gradient_polish(params, start_time, max_iter=40)
        params = _fix_validity(params)
        
        score, _ = _eval_packing(params)
        if score > best_score:
            best_score = score
            best_params = params
    
    # Final validity fix
    best_params = _fix_validity(best_params)
    
    return best_params
>>>>>>> REPLACE
