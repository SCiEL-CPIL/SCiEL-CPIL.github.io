"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _objective_and_grad(params, n):
    """
    Objective: minimize -sum(radii) + penalty(constraints)
    params: (2n, n) -> positions (x,y) for n circles, radii r_i
    """
    pos = params[:2*n].reshape(n, 2)
    log_r = params[2*n:]
    r = np.exp(log_r)
    
    # Base objective: maximize sum of radii (minimize negative sum)
    obj = -np.sum(r)
    grad_obj = np.zeros_like(params)
    grad_obj[2*n:] = -r  # d(-sum(r))/d(log_r) = -r
    
    # Penalty for boundary violations
    # Constraints: x_i >= r_i, x_i <= 1-r_i, y_i >= r_i, y_i <= 1-r_i
    # Smooth penalty: sum(max(0, violation)^2)
    boundary_violations = np.maximum(0, r - pos[:, 0]) + \
                          np.maximum(0, r - pos[:, 1]) + \
                          np.maximum(0, r - (1 - pos[:, 0])) + \
                          np.maximum(0, r - (1 - pos[:, 1]))
    
    penalty = np.sum(boundary_violations**2)
    obj += penalty
    
    # Gradient of penalty
    # d/dx max(0, r-x) = -1 if r-x > 0 else 0
    grad_pen_x = np.zeros_like(pos)
    grad_pen_y = np.zeros_like(pos)
    
    mask_x_left = boundary_violations > 0 # This is too rough, need specific terms
    
    # Let's compute gradients properly for each term
    # Term 1: max(0, r - x)^2 -> 2*max(0, r-x)*(1 if r-x>0 else 0) * d(r-x)/dx = -2*max(0, r-x)
    # Term 2: max(0, r - y)^2 -> -2*max(0, r-y)
    # Term 3: max(0, r - (1-x))^2 = max(0, r-1+x)^2 -> 2*max(0, r-1+x)
    # Term 4: max(0, r - (1-y))^2 = max(0, r-1+y)^2 -> 2*max(0, r-1+y)
    
    v1 = np.maximum(0, r - pos[:, 0])
    v2 = np.maximum(0, r - pos[:, 1])
    v3 = np.maximum(0, r - (1 - pos[:, 0]))
    v4 = np.maximum(0, r - (1 - pos[:, 1]))
    
    grad_pen_x[:, 0] = -2*v1 + 2*v3
    grad_pen_y[:, 0] = -2*v2 + 2*v4
    
    # Gradient w.r.t log_r: d(r)/d(log_r) = r
    # d/d(log_r) (v1^2) = 2*v1 * d(r-x)/d(log_r) = 2*v1 * r
    grad_pen_r = 2*r*(v1 + v2 + v3 + v4)
    
    grad_obj[0:n] += grad_pen_x[:, 0]
    grad_obj[n:2*n] += grad_pen_y[:, 0]
    grad_obj[2*n:] += grad_pen_r
    
    # Penalty for circle-circle overlaps
    # dist_ij < r_i + r_j -> violation = (r_i + r_j) - dist_ij
    # Penalty: sum(violation^2)
    
    # Vectorized distance computation
    # pos shape (n, 2)
    # We need upper triangle indices
    idx_i, idx_j = np.triu_indices(n, k=1)
    diff = pos[idx_i] - pos[idx_j]
    dist = np.linalg.norm(diff, axis=1)
    
    r_sum = r[idx_i] + r[idx_j]
    viol = np.maximum(0, r_sum - dist)
    
    penalty_cc = np.sum(viol**2)
    obj += penalty_cc
    
    # Gradient of penalty_cc
    # d/dx_i sum_k max(0, r_ik - dist_ik)^2
    # For each pair (i, j):
    # d/dx_i (r_i + r_j - dist_ij)^2 = 2*(r_i+r_j-dist_ij) * (-d(dist_ij)/dx_i)
    # d(dist_ij)/dx_i = (x_i - x_j) / dist_ij
    # So grad_x_i += -2*viol * (x_i - x_j)/dist_ij = 2*viol * (x_j - x_i)/dist_ij
    
    grad_pos = np.zeros((n, 2))
    grad_r = np.zeros(n)
    
    # Avoid division by zero
    safe_dist = np.where(dist > 1e-8, dist, 1e-8)
    
    # Only contribute if violation > 0
    active = viol > 0
    if np.any(active):
        active_idx_i = idx_i[active]
        active_idx_j = idx_j[active]
        active_viol = viol[active]
        active_dist = safe_dist[active]
        active_diff = diff[active] # pos[i] - pos[j]
        
        # grad for pos[i]
        # term = 2 * viol * (pos[j] - pos[i]) / dist
        # = -2 * viol * active_diff / active_dist
        grad_contrib = -2 * active_viol[:, None] * active_diff / active_dist[:, None]
        
        # Accumulate for i
        np.add.at(grad_pos, active_idx_i, grad_contrib)
        # Accumulate for j (opposite direction)
        np.add.at(grad_pos, active_idx_j, -grad_contrib)
        
        # grad for r_i and r_j
        # d/d(r_i) (r_i + r_j - dist)^2 = 2*viol
        np.add.at(grad_r, active_idx_i, 2*active_viol)
        np.add.at(grad_r, active_idx_j, 2*active_viol)
        
        # Convert grad_r (w.r.t r) to w.r.t log_r: grad_log_r = grad_r * r
        grad_log_r = grad_r * r
        
        grad_obj[0:n] += grad_pos[:, 0]
        grad_obj[n:2*n] += grad_pos[:, 1]
        grad_obj[2*n:] += grad_log_r
    
    return obj, grad_obj

def _is_valid(circles, tol=1e-9):
    n = len(circles)
    for i in range(n):
        x, y, r = circles[i]
        if r <= 0:
            return False
        if x < r - tol or x > 1 - r + tol or y < r - tol or y > 1 - r + tol:
            return False
    for i in range(n):
        for j in range(i + 1, n):
            dx = circles[i][0] - circles[j][0]
            dy = circles[i][1] - circles[j][1]
            dist = np.sqrt(dx*dx + dy*dy)
            if dist < circles[i][2] + circles[j][2] - tol:
                return False
    return True

def _compute_score(circles):
    if not _is_valid(circles):
        return -1.0
    return np.sum(circles[:, 2])

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    n = N
    best_score = -1.0
    best_circles = None
    
    # Multi-start with different initial configurations
    starts = []
    
    # 1. Hexagonal grid
    # Approximate hex grid for 26 circles
    # 5 rows: 5, 5, 5, 5, 6? Or 4 rows: 6, 6, 6, 6, 2?
    # Let's try a dense hex packing scaled to fit.
    # Ideal hex packing density ~0.9069.
    # Area per circle ~ pi*r^2. Total area ~ n*pi*r^2.
    # For n=26, if r=0.1, area ~ 0.81. 
    # Let's generate a hex grid and scale it.
    
    def hex_grid(n_circles):
        # Generate points in hex lattice
        points = []
        # Number of rows
        # Try to make it roughly square
        # 5x5 = 25, add 1
        # Let's just generate a large hex grid and pick the first n
        # Or generate row by row
        # Row 0: 5, Row 1: 5, Row 2: 5, Row 3: 5, Row 4: 6
        rows = [5, 5, 5, 5, 6]
        spacing = 0.18 # Will scale later
        for i, count in enumerate(rows):
            offset = (i % 2) * spacing * 0.5
            for j in range(count):
                x = j * spacing + offset
                y = i * spacing * np.sqrt(3)/2
                points.append([x, y])
        points = np.array(points[:n_circles])
        
        # Center and scale to fit in [0,1]
        # Find bounding box
        min_x, min_y = points.min(axis=0)
        max_x, max_y = points.max(axis=0)
        
        # Scale such that max distance from center + radius <= 0.5
        # Approx radius
        r_est = 0.08
        # Scale factor
        # We want the set of points to fit in [r, 1-r]
        # Let's just normalize to [0,1] then shrink
        points = (points - min_x) / (max_x - min_x + 1e-9)
        points = (points - min_y) / (max_y - min_y + 1e-9)
        
        # This normalization distorts aspect ratio. Let's do uniform scaling.
        points_raw = np.array(points)
        # Re-do:
        pts = []
        for i, count in enumerate(rows):
            offset = (i % 2) * 0.5
            for j in range(count):
                x = j + offset
                y = i * np.sqrt(3)/2
                pts.append([x, y])
        pts = np.array(pts[:n_circles])
        
        min_pt = pts.min(axis=0)
        max_pt = pts.max(axis=0)
        extent = max(max_pt[0] - min_pt[0], max_pt[1] - min_pt[1])
        
        # Scale to fit in square of side 0.8 (leaving 0.1 margin)
        scale = 0.8 / extent
        pts = (pts - min_pt) * scale
        pts += 0.1
        
        return pts, 0.08 # Initial radius guess
    
    pts_hex, r_hex = hex_grid(n)
    starts.append((pts_hex, r_hex))
    
    # 2. Square grid
    side = int(np.ceil(np.sqrt(n)))
    pts_sq = np.array([[(i + 0.5)/side, (j + 0.5)/side] for i in range(side) for j in range(side)])[:n]
    starts.append((pts_sq, 0.5/side - 0.01))
    
    # 3. Random
    np.random.seed(42)
    pts_rand = np.random.rand(n, 2) * 0.6 + 0.2
    starts.append((pts_rand, 0.08))
    
    for idx, (init_pos, init_r) in enumerate(starts):
        if time.time() > deadline:
            break
            
        # Initial params
        # pos: (n, 2) -> flat (2n)
        # log_r: (n,)
        init_params = np.concatenate([init_pos.flatten(), np.log(init_r * np.ones(n))])
        
        # Bounds
        # x, y in [0, 1] (loose, penalty handles tight constraints)
        # log_r in [log(0.01), log(0.5)]
        bounds = [(0.0, 1.0)] * (2*n) + [(np.log(0.01), np.log(0.5))] * n
        
        # Run optimization
        try:
            res = minimize(
                _objective_and_grad, 
                init_params, 
                method='L-BFGS-B', 
                bounds=bounds,
                jac=True,
                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
            )
        except Exception:
            continue
            
        if res.success or res.status in [0, 1, 2, 3, 4]: # 0: maxiter, 1: ftol, etc.
            opt_params = res.x
            opt_pos = opt_params[:2*n].reshape(n, 2)
            opt_log_r = opt_params[2*n:]
            opt_r = np.exp(opt_log_r)
            
            # Clip positions to [0,1] just in case
            opt_pos = np.clip(opt_pos, 0, 1)
            
            circles = np.column_stack([opt_pos, opt_r])
            
            # Validate
            score = _compute_score(circles)
            if score > best_score:
                best_score = score
                best_circles = circles.copy()
                
                # If we have a very good score, we might stop early or try to refine
                # But let's try all starts if time permits.
    
    # Final check
    if best_circles is None:
        # Fallback to simple grid
        side = int(np.ceil(np.sqrt(n)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 0.01) for i in range(side) for j in range(side)]
        best_circles = np.array(cells[:n])
        
    return best_circles
# EVOLVE-BLOCK-END
