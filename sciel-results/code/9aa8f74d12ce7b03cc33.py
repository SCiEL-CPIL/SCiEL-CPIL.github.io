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

def _max_radius_at(pos, others):
    """
    Given a position pos=(x,y) and a list of other circles (x,y,r),
    return the maximum radius a circle at pos can have without overlapping
    others or going out of bounds.
    """
    x, y = pos
    # Boundary constraint
    r_max = min(x, 1.0 - x, y, 1.0 - y)
    if r_max < 0:
        return 0.0
    # Overlap constraint with each other circle
    for ox, oy, orad in others:
        d = np.sqrt((x - ox)**2 + (y - oy)**2)
        avail = d - orad
        if avail < r_max:
            r_max = avail
    return max(0.0, r_max)

def _best_fit_circle(params, i, grid_res=40):
    """
    For circle i, find the best position (and resulting max radius) by
    scanning a local grid around its current position.
    Returns (best_x, best_y, best_r, improvement) where improvement = new_r - old_r.
    """
    x0, y0, r0 = params[i]
    others = [(params[j, 0], params[j, 1], params[j, 2]) for j in range(N) if j != i]
    
    # Search region: expand around current position
    # The circle can move within bounds
    search_radius = max(0.05, r0 * 1.5)
    
    x_min = max(0.0, x0 - search_radius)
    x_max = min(1.0, x0 + search_radius)
    y_min = max(0.0, y0 - search_radius)
    y_max = min(1.0, y0 + search_radius)
    
    best_r = r0
    best_x, best_y = x0, y0
    
    # Coarse grid search
    xs = np.linspace(x_min, x_max, grid_res)
    ys = np.linspace(y_min, y_max, grid_res)
    
    for x in xs:
        for y in ys:
            r = _max_radius_at((x, y), others)
            if r > best_r:
                best_r = r
                best_x, best_y = x, y
    
    # Fine grid search around the best coarse point
    fine_res = 20
    fine_radius = search_radius / grid_res * 1.5
    fx_min = max(0.0, best_x - fine_radius)
    fx_max = min(1.0, best_x + fine_radius)
    fy_min = max(0.0, best_y - fine_radius)
    fy_max = min(1.0, best_y + fine_radius)
    
    fxs = np.linspace(fx_min, fx_max, fine_res)
    fys = np.linspace(fy_min, fy_max, fine_res)
    
    for x in fxs:
        for y in fys:
            r = _max_radius_at((x, y), others)
            if r > best_r:
                best_r = r
                best_x, best_y = x, y
    
    improvement = best_r - r0
    return best_x, best_y, best_r, improvement

def _is_valid(params, tol=1e-6):
    """Check if all circles are valid (inside unit square, no overlaps)."""
    for i in range(N):
        x, y, r = params[i]
        if x < r - tol or x > 1.0 - r + tol or y < r - tol or y > 1.0 - r + tol:
            return False
    for i in range(N):
        for j in range(i + 1, N):
            dx = params[i, 0] - params[j, 0]
            dy = params[i, 1] - params[j, 1]
            d = np.sqrt(dx*dx + dy*dy)
            if d < params[i, 2] + params[j, 2] - tol:
                return False
    return True

def _score(params):
    """Return sum of radii if valid, else -inf."""
    if not _is_valid(params):
        return -np.inf
    return np.sum(params[:, 2])

def _grid_init(n=N):
    """Create an initial grid packing."""
    side = int(np.ceil(np.sqrt(n)))
    cells = []
    for i in range(side):
        for j in range(side):
            if len(cells) < n:
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                r = 0.5 / side - 1e-4
                cells.append([x, y, r])
    return np.array(cells[:n])

def _random_init(n=N, seed=42):
    """Create a random initial packing with small circles."""
    rng = np.random.default_rng(seed)
    params = np.zeros((n, 3))
    for i in range(n):
        r = rng.uniform(0.02, 0.08)
        x = rng.uniform(r, 1.0 - r)
        y = rng.uniform(r, 1.0 - r)
        params[i] = [x, y, r]
    return params

def _best_fit_local_search(params, max_rounds=50, time_budget=None, start_time=None):
    """
    Iteratively improve the packing using best-fit local search.
    For each circle, find the position that maximizes its radius given others.
    """
    params = params.copy()
    current_score = _score(params)
    
    if current_score == -np.inf:
        # Fix invalid packing first
        for i in range(N):
            x, y, r = params[i]
            r = min(r, x, 1.0-x, y, 1.0-y)
            params[i, 2] = max(0.01, r)
        current_score = _score(params)
    
    no_improve_count = 0
    
    for round_num in range(max_rounds):
        if time_budget is not None and start_time is not None:
            if time.time() - start_time > time_budget:
                break
        
        improved = False
        for i in range(N):
            if time_budget is not None and start_time is not None:
                if time.time() - start_time > time_budget:
                    break
            
            best_x, best_y, best_r, improvement = _best_fit_circle(params, i)
            
            if improvement > 1e-8:
                params[i] = [best_x, best_y, best_r]
                improved = True
        
        if not improved:
            no_improve_count += 1
            if no_improve_count >= 3:
                break
        else:
            no_improve_count = 0
    
    return params

def _fix_packing(params):
    """Ensure the packing is valid by shrinking radii as needed."""
    params = params.copy()
    for _ in range(20):
        changed = False
        # Fix boundaries
        for i in range(N):
            x, y, r = params[i]
            max_r = min(x, 1.0-x, y, 1.0-y)
            if r > max_r:
                params[i, 2] = max(0.0, max_r)
                changed = True
        
        # Fix overlaps
        for i in range(N):
            for j in range(i + 1, N):
                dx = params[i, 0] - params[j, 0]
                dy = params[i, 1] - params[j, 1]
                d = np.sqrt(dx*dx + dy*dy)
                required = params[i, 2] + params[j, 2]
                if d < required:
                    # Shrink the smaller circle
                    if params[i, 2] <= params[j, 2]:
                        new_r = max(0.0, d - params[j, 2])
                        params[i, 2] = new_r
                    else:
                        new_r = max(0.0, d - params[i, 2])
                        params[j, 2] = new_r
                    changed = True
        if not changed:
            break
    return params

def construct():
    start_time = time.time()
    timeout = 130.0  # Safety margin (evaluator kills at 150s)
    
    # Try multiple initializations and local searches
    best_params = None
    best_score = -np.inf
    
    rng = np.random.default_rng(42)
    
    # Strategy 1: Grid initialization
    init1 = _grid_init(N)
    best_params = _best_fit_local_search(init1, max_rounds=100, time_budget=timeout, start_time=start_time)
    best_score = _score(best_params)
    
    # Strategy 2: Random initialization
    if time.time() - start_time < timeout - 10:
        init2 = _random_init(N, seed=123)
        candidate = _best_fit_local_search(init2, max_rounds=100, time_budget=timeout, start_time=start_time)
        cand_score = _score(candidate)
        if cand_score > best_score:
            best_score = cand_score
            best_params = candidate
    
    # Strategy 3: Perturb best and re-optimize
    if time.time() - start_time < timeout - 10 and best_params is not None:
        for attempt in range(5):
            if time.time() - start_time > timeout - 5:
                break
            perturbed = best_params.copy()
            noise = 0.03
            perturbed[:, 0] += rng.uniform(-noise, noise, N)
            perturbed[:, 1] += rng.uniform(-noise, noise, N)
            perturbed[:, 0] = np.clip(perturbed[:, 0], 0.01, 0.99)
            perturbed[:, 1] = np.clip(perturbed[:, 1], 0.01, 0.99)
            
            candidate = _best_fit_local_search(perturbed, max_rounds=50, time_budget=timeout, start_time=start_time)
            cand_score = _score(candidate)
            if cand_score > best_score:
                best_score = cand_score
                best_params = candidate
    
    # Final fix to ensure validity
    if best_params is not None:
        best_params = _fix_packing(best_params)
        # Re-check score after fix
        final_score = _score(best_params)
        if final_score == -np.inf:
            # Fallback to grid
            best_params = _fix_packing(_grid_init(N))
    
    # Ensure we return a valid result
    if best_params is None or not _is_valid(best_params):
        best_params = _fix_packing(_grid_init(N))
    
    return best_params
>>>>>>> REPLACE
