"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time

N = 21

# EVOLVE-BLOCK-START
def _soft_penalty(violation, scale=100.0):
    """Smooth penalty for constraint violation."""
    return scale * np.logaddexp(0, violation / 0.01)

def _vectorized_objective(x, w, h, penalty_scale=100.0):
    """
    Vectorized objective with soft penalties.
    x: array of shape (3*N,) with [cx, cy, r] for each circle
    Returns negative sum of radii (for minimization) with penalties.
    """
    cx = x[0::3]
    cy = x[1::3]
    r = x[2::3]
    
    # Sum of radii (to maximize, so minimize negative)
    total_r = np.sum(r)
    
    # Boundary violations
    left_viol = np.maximum(0, r - cx)
    right_viol = np.maximum(0, cx + r - w)
    bottom_viol = np.maximum(0, r - cy)
    top_viol = np.maximum(0, cy + r - h)
    
    boundary_penalty = np.sum([
        _soft_penalty(left_viol, penalty_scale),
        _soft_penalty(right_viol, penalty_scale),
        _soft_penalty(bottom_viol, penalty_scale),
        _soft_penalty(top_viol, penalty_scale)
    ])
    
    # Overlap violations - compute all pair distances
    # Use broadcasting for efficiency
    cx_diff = cx[:, None] - cx[None, :]
    cy_diff = cy[:, None] - cy[None, :]
    r_sum = r[:, None] + r[None, :]
    
    dist = np.sqrt(cx_diff**2 + cy_diff**2)
    overlap = np.maximum(0, r_sum - dist)
    
    # Only count upper triangle to avoid double counting
    triu_mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    overlap_penalty = _soft_penalty(overlap[triu_mask], penalty_scale)
    
    # Negative radius penalty
    neg_r_penalty = _soft_penalty(np.maximum(0, -r), penalty_scale)
    
    return -total_r + boundary_penalty + overlap_penalty + neg_r_penalty

def _check_feasibility(circles, w, h, tol=1e-8):
    """Check if circles are feasible."""
    for cx, cy, r in circles:
        if r < -tol or cx - r < -tol or cx + r > w + tol or cy - r < -tol or cy + r > h + tol:
            return False
    for i in range(N):
        for j in range(i+1, N):
            dx = circles[i][0] - circles[j][0]
            dy = circles[i][1] - circles[j][1]
            dist = np.sqrt(dx*dx + dy*dy)
            if dist < circles[i][2] + circles[j][2] - tol:
                return False
    return True

def _total_radius(circles):
    return sum(c[2] for c in circles)

def _grid_init(w, h, n_circles=N, staggered=True):
    """Generate initial grid positions."""
    circles = []
    # Try different grid configurations
    for rows in range(3, 8):
        cols = int(np.ceil(n_circles / rows))
        if rows * cols < n_circles:
            continue
        
        r_est = min(w / cols, h / rows) * 0.45
        for i in range(rows):
            for j in range(cols):
                if len(circles) >= n_circles:
                    break
                cx = (j + 0.5) / cols * w
                cy = (i + 0.5) / rows * h
                if staggered and i % 2 == 1:
                    cx += 0.3 / cols * w
                r = min(r_est, min(cx, w - cx, cy, h - cy) * 0.9)
                r = max(r, 0.01)
                circles.append((cx, cy, r))
        
        if len(circles) == n_circles:
            break
    
    # Fill remaining if needed
    while len(circles) < n_circles:
        circles.append((w/2, h/2, 0.01))
    
    return circles[:n_circles]

def construct():
    start_time = time.time()
    deadline = 145.0  # safety margin before 150s limit
    
    best_result = None
    best_score = 0
    
    # Try multiple aspect ratios
    aspect_ratios = [1.0, 0.8, 1.25, 0.7, 1.4, 0.6, 1.6]
    
    for aspect in aspect_ratios:
        if time.time() > deadline:
            break
        
        w = 2.0 * aspect / (1 + aspect)
        h = 2.0 / (1 + aspect)
        
        # Generate initial grid
        initial_circles = _grid_init(w, h, staggered=True)
        
        # Convert to vector
        x0 = np.array([v for c in initial_circles for v in c])
        
        # Bounds
        bounds = []
        for _ in range(N):
            bounds.append((0.01, w - 0.01))  # cx
            bounds.append((0.01, h - 0.01))  # cy
            bounds.append((0.01, min(w, h) * 0.45))  # r
        
        # Try multiple optimization methods and restarts
        for method_name, method, opts in [
            ('SLSQP', 'SLSQP', {'maxiter': 200, 'ftol': 1e-8}),
            ('Nelder-Mead', 'Nelder-Mead', {'maxiter': 500, 'xatol': 1e-7, 'fatol': 1e-7}),
        ]:
            if time.time() > deadline:
                break
            
            for restart in range(3):
                if time.time() > deadline:
                    break
                
                if restart == 0:
                    x_init = x0.copy()
                else:
                    # Perturb initial solution
                    x_init = x0 + np.random.normal(0, 0.02, size=x0.shape)
                    x_init = np.clip(x_init, 
                                     [b[0] for b in bounds],
                                     [b[1] for b in bounds])
                
                try:
                    res = minimize(
                        lambda x: _vectorized_objective(x, w, h),
                        x_init,
                        method=method,
                        options=opts
                    )
                    
                    # Extract circles from result
                    circles = []
                    for i in range(N):
                        cx = max(0.001, min(res.x[i*3], w - 0.001))
                        cy = max(0.001, min(res.x[i*3+1], h - 0.001))
                        r = max(0.001, min(res.x[i*3+2], min(w, h) * 0.45))
                        circles.append((cx, cy, r))
                    
                    # Check feasibility
                    if _check_feasibility(circles, w, h):
                        score = _total_radius(circles)
                        if score > best_score:
                            best_score = score
                            best_result = circles
                            
                except Exception:
                    pass
    
    # If we found a good result, return it
    if best_result is not None:
        return np.array(best_result)
    
    # Fallback to simple grid in 1x1 square
    fallback = _grid_init(1.0, 1.0, staggered=False)
    return np.array(fallback)
# EVOLVE-BLOCK-END
