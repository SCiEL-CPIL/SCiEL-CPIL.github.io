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
    if np.any(rs < -tol):
        return False
    
    # Check bounds
    # The bounding box of the circles must fit in w x h
    # However, the problem says "bounding rectangle has perimeter at most 4".
    # This implies we can choose the rectangle size, or rather, the circles define the rectangle.
    # Wait, the interface says: "n disjoint circles whose bounding rectangle has perimeter at most 4".
    # This means the evaluator calculates the bounding box of the circles and checks its perimeter.
    # It does NOT mean the circles are constrained to a fixed [0,w]x[0,h] box provided by us.
    # BUT, the current code and inspiration code assume a fixed box [0,w]x[0,h] where w+h=2.
    # If we pack circles in [0,w]x[0,h], their bounding box is at most w x h.
    # Perimeter = 2(w+h) = 4. This is valid.
    # So, packing in a box of perimeter 4 is a valid strategy.
    
    # Check if circles are inside [0, w] x [0, h]
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

def _resolve_collisions(circles, w, h, max_iters=1000, tol=1e-5):
    """
    Relaxation algorithm to resolve overlaps and boundary violations.
    circles: (n, 3) array [x, y, r]
    Returns modified circles array.
    """
    n = circles.shape[0]
    circles = circles.copy()
    
    for _ in range(max_iters):
        moved = False
        
        # 1. Boundary constraints: push circles inside [0, w] x [0, h]
        for i in range(n):
            x, y, r = circles[i]
            if x - r < 0:
                circles[i, 0] = r
                moved = True
            if x + r > w:
                circles[i, 0] = w - r
                moved = True
            if y - r < 0:
                circles[i, 1] = r
                moved = True
            if y + r > h:
                circles[i, 1] = h - r
                moved = True
        
        # 2. Pairwise collisions: push apart
        for i in range(n):
            for j in range(i+1, n):
                dx = circles[j, 0] - circles[i, 0]
                dy = circles[j, 1] - circles[i, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                min_dist = circles[i, 2] + circles[j, 2]
                
                if dist < min_dist:
                    if dist < 1e-8:
                        # Random perturbation if exactly overlapping
                        circles[i, 0] += np.random.uniform(-0.01, 0.01)
                        circles[i, 1] += np.random.uniform(-0.01, 0.01)
                        circles[j, 0] -= np.random.uniform(-0.01, 0.01)
                        circles[j, 1] -= np.random.uniform(-0.01, 0.01)
                        moved = True
                    else:
                        overlap = min_dist - dist
                        nx = dx / dist
                        ny = dy / dist
                        # Push apart by half the overlap each
                        circles[i, 0] -= nx * overlap * 0.5
                        circles[i, 1] -= ny * overlap * 0.5
                        circles[j, 0] += nx * overlap * 0.5
                        circles[j, 1] += ny * overlap * 0.5
                        moved = True
        
        if not moved:
            break
            
    return circles

def _greedy_pack(w, h, init_circles, time_limit):
    """
    Greedy radius growth strategy.
    Starts with init_circles, increases radii uniformly, resolves collisions.
    """
    start_time = time.time()
    circles = init_circles.copy()
    current_min_r = np.min(circles[:, 2])
    growth_factor = 1.005 # 0.5% growth per step
    max_growth_steps = 500
    
    for step in range(max_growth_steps):
        if time.time() - start_time > time_limit:
            break
            
        # Increase radii
        new_radii = circles[:, 2] * growth_factor
        
        # Optional: Cap radii to avoid ridiculous sizes
        max_possible_r = min(w, h) / 2.0
        new_radii = np.minimum(new_radii, max_possible_r)
        
        circles[:, 2] = new_radii
        
        # Resolve collisions
        circles = _resolve_collisions(circles, w, h, max_iters=200)
        
        # Check if valid
        if _validate_solution(circles, w, h, N):
            current_min_r = np.min(circles[:, 2])
        else:
            # Rollback radii if invalid
            circles[:, 2] = circles[:, 2] / growth_factor
            # Try a smaller growth factor for next step? Or break?
            # If we can't grow even slightly, we are likely at a local optimum for this config.
            # Let's try one more time with a tiny growth to see if it's numerical noise.
            circles[:, 2] *= 1.001
            circles = _resolve_collisions(circles, w, h, max_iters=50)
            if not _validate_solution(circles, w, h, N):
                circles[:, 2] /= 1.001
                break
                
    return circles

def _refine_with_scipy(pos_r, w, h, time_limit):
    """
    Use scipy L-BFGS-B to refine positions and radii.
    Objective: maximize sum(r) subject to constraints.
    We use a penalty method to handle constraints.
    """
    start_time = time.time()
    n_circles = N
    n_vars = n_circles * 3
    
    # Initial vector
    x0 = pos_r.flatten()
    
    # Bounds: x in [0, w], y in [0, h], r in [0, min(w,h)/2]
    max_r = min(w, h) / 2.0
    bounds = []
    for i in range(n_circles):
        bounds.append((0, w))
        bounds.append((0, h))
        bounds.append((0, max_r))
        
    def objective_and_grad(x):
        x_coords = x[0:n_circles*2:2]
        y_coords = x[1:n_circles*2:2]
        r_coords = x[2:n_circles*2:2]
        
        val = -np.sum(r_coords)
        grad = np.zeros_like(x)
        grad[2:n_circles*2:2] = -1.0
        
        penalty_factor = 1000.0
        
        # Pairwise constraints
        for i in range(n_circles):
            for j in range(i+1, n_circles):
                dx = x_coords[i] - x_coords[j]
                dy = y_coords[i] - y_coords[j]
                dist = np.sqrt(dx*dx + dy*dy)
                
                if dist < 1e-8:
                    dist = 1e-8
                    dx = 1e-8
                    dy = 0.0
                
                sum_r = r_coords[i] + r_coords[j]
                violation = sum_r - dist
                
                if violation > 0:
                    val += penalty_factor * violation**2
                    g_penalty = 2.0 * penalty_factor * violation
                    grad[2*i] += -g_penalty * (dx / dist)
                    grad[2*i+1] += -g_penalty * (dy / dist)
                    grad[2*i+2] += g_penalty
                    grad[2*j] += g_penalty * (dx / dist)
                    grad[2*j+1] += g_penalty * (dy / dist)
                    grad[2*j+2] += g_penalty
        
        # Wall constraints
        for i in range(n_circles):
            r = r_coords[i]
            x = x_coords[i]
            y = y_coords[i]
            
            v = r - x
            if v > 0:
                val += penalty_factor * v**2
                grad[2*i] += -2.0 * penalty_factor * v
                grad[2*i+2] += 2.0 * penalty_factor * v
                
            v = r - (w - x)
            if v > 0:
                val += penalty_factor * v**2
                grad[2*i] += 2.0 * penalty_factor * v
                grad[2*i+2] += 2.0 * penalty_factor * v
                
            v = r - y
            if v > 0:
                val += penalty_factor * v**2
                grad[2*i+1] += -2.0 * penalty_factor * v
                grad[2*i+2] += 2.0 * penalty_factor * v
                
            v = r - (h - y)
            if v > 0:
                val += penalty_factor * v**2
                grad[2*i+1] += 2.0 * penalty_factor * v
                grad[2*i+2] += 2.0 * penalty_factor * v
                
        return val, grad

    def callback(Xk):
        return time.time() - start_time > time_limit

    try:
        res = minimize(
            lambda x: objective_and_grad(x)[0],
            x0,
            jac=lambda x: objective_and_grad(x)[1],
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 200, 'ftol': 1e-12, 'maxfun': 500, 'callback': callback}
        )
        
        pos_r_refined = res.x.reshape(-1, 3)
        
        # Validate refined solution
        if _validate_solution(pos_r_refined, w, h, N):
            return pos_r_refined
        else:
            # Try to fix small violations by shrinking radii
            # Simple fix: ensure radii are small enough to fit
            for i in range(N):
                x, y, r = pos_r_refined[i]
                min_wall_dist = min(x, w-x, y, h-y)
                pos_r_refined[i, 2] = min(r, min_wall_dist - 1e-6)
            # Re-check overlaps? If still invalid, return original
            if _validate_solution(pos_r_refined, w, h, N):
                return pos_r_refined
            else:
                return pos_r # Return original if refinement failed

    except Exception:
        return pos_r

def _generate_grid_init(w, h, n_circles, cols, rows):
    """
    Generate initial circle positions in a grid pattern.
    """
    circles = np.zeros((n_circles, 3))
    count = 0
    for i in range(cols):
        for j in range(rows):
            if count >= n_circles:
                break
            x = (i + 0.5) * w / cols
            y = (j + 0.5) * h / rows
            # Initial radius: half the min cell size, slightly smaller
            cell_w = w / cols
            cell_h = h / rows
            r = 0.45 * min(cell_w, cell_h)
            
            circles[count, 0] = x
            circles[count, 1] = y
            circles[count, 2] = r
            count += 1
            
    return circles

def construct():
    start_time = time.time()
    deadline = start_time + 115.0 # 5s margin
    
    best_result = None
    best_score = 0.0
    
    # Aspect ratios to try: w/h
    # w + h = 2 => w = 2*ar/(1+ar), h = 2/(1+ar)
    aspect_ratios = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5]
    
    # Grid configurations to try for each aspect ratio
    # (cols, rows) such that cols*rows >= 21
    grid_configs = [
        (4, 5), (5, 4),
        (3, 7), (7, 3),
        (3, 8), (8, 3),
        (4, 6), (6, 4),
        (5, 5)
    ]
    
    candidates = []
    
    for ar in aspect_ratios:
        if time.time() > deadline - 30: # Leave time for refinement
            break
            
        w = 2.0 * ar / (1 + ar)
        h = 2.0 / (1 + ar)
        
        for cols, rows in grid_configs:
            if cols * rows < N:
                continue
                
            # Check if aspect ratio of grid matches aspect ratio of box roughly
            grid_ar = cols / rows
            box_ar = w / h
            if abs(grid_ar - box_ar) / box_ar > 0.5:
                continue
                
            init_circles = _generate_grid_init(w, h, N, cols, rows)
            
            # Time budget for greedy growth per config
            time_left = deadline - 30 - time.time()
            if time_left < 2:
                break
                
            greedy_time = min(5.0, time_left / 10.0) # Limit greedy to 5s max per config
            
            result = _greedy_pack(w, h, init_circles, greedy_time)
            
            if _validate_solution(result, w, h, N):
                score = np.sum(result[:, 2])
                if score > best_score:
                    best_score = score
                    best_result = result
                candidates.append((score, result, w, h))
                
            if time.time() > deadline - 30:
                break
                
    # Refine top candidates
    if candidates:
        # Sort by score descending
        candidates.sort(key=lambda x: x[0], reverse=True)
        
        for score, pos_r, w, h in candidates[:3]: # Refine top 3
            if time.time() > deadline - 5:
                break
                
            refine_time = min(15.0, deadline - 5 - time.time())
            if refine_time < 1:
                break
                
            refined = _refine_with_scipy(pos_r, w, h, refine_time)
            
            if _validate_solution(refined, w, h, N):
                new_score = np.sum(refined[:, 2])
                if new_score > best_score:
                    best_score = new_score
                    best_result = refined

    # Fallback
    if best_result is None:
        w, h = 1.0, 1.0
        cols, rows = 5, 5
        best_result = _generate_grid_init(w, h, N, cols, rows)
        # Ensure it's valid by shrinking radii if needed
        for i in range(N):
            x, y, r = best_result[i]
            min_d = min(x, w-x, y, h-y)
            best_result[i, 2] = min(r, min_d - 1e-4)
        # Simple relaxation to fix overlaps
        best_result = _resolve_collisions(best_result, w, h, max_iters=100)
        
    return best_result
