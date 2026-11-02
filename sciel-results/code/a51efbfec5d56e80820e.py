"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from scipy.optimize import minimize
from scipy.special import logsumexp
from scipy.spatial import distance

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # 20s safety margin
    
    # Pre-compute pairwise indices for vectorized distance calculation
    i_idx, j_idx = np.triu_indices(N, k=1)
    pairs = np.stack([i_idx, j_idx], axis=1) # shape (M, 2)
    
    def objective(x, aspect):
        """
        x: flattened array of [x1, y1, r1, x2, y2, r2, ...]
        aspect: width/height ratio
        Returns: Negative of (sum_radii - penalty)
        We want to maximize sum_radii - penalty.
        """
        w = 2.0 * aspect / (1.0 + aspect)
        h = 2.0 / (1.0 + aspect)
        
        xy = x[0::3]
        yv = x[1::3]
        r = x[2::3]
        
        # 1. Boundary Penalties
        # Violations: x < r, x > w-r, y < r, y > h-r
        # We want these to be <= 0. Violation is max(0, val).
        # Soft penalty: sum(softplus(val))
        
        # x_min violations: x - r < 0 is good. x - r > 0 is bad.
        # Actually, constraint is x >= r. Violation = r - x.
        # Let's define violation as amount we are OUTSIDE.
        # x < r -> violation = r - x
        # x > w - r -> violation = x - (w - r)
        
        v_x_min = r - xy
        v_x_max = xy - (w - r)
        v_y_min = r - yv
        v_y_max = yv - (h - r)
        
        # Only count positive violations
        # Use softplus for differentiability: log(1 + exp(v)) / beta
        # For large v, this approx v. For small v, it's smooth.
        beta = 50.0
        pen_x_min = np.log1p(np.exp(beta * v_x_min)) / beta
        pen_x_max = np.log1p(np.exp(beta * v_x_max)) / beta
        pen_y_min = np.log1p(np.exp(beta * v_y_min)) / beta
        pen_y_max = np.log1p(np.exp(beta * v_y_max)) / beta
        
        boundary_penalty = np.sum(pen_x_min) + np.sum(pen_x_max) + np.sum(pen_y_min) + np.sum(pen_y_max)
        
        # 2. Pairwise Collision Penalties
        # dist < r_i + r_j is violation.
        # violation = (r_i + r_j) - dist
        dx = xy[pairs[:, 0]] - xy[pairs[:, 1]]
        dy = yv[pairs[:, 0]] - yv[pairs[:, 1]]
        dist = np.sqrt(dx*dx + dy*dy + 1e-12)
        
        r_sum = r[pairs[:, 0]] + r[pairs[:, 1]]
        v_coll = r_sum - dist
        
        pen_coll = np.sum(np.log1p(np.exp(beta * v_coll)) / beta)
        
        total_penalty = boundary_penalty + pen_coll
        
        # Objective: Maximize sum(r) - Penalty
        # Minimize: -(sum(r) - Penalty) = Penalty - sum(r)
        return total_penalty - np.sum(r)

    def run_nelder_mead(x0, aspect, max_iter=5000):
        """
        Runs Nelder-Mead on the fixed aspect ratio.
        Returns best_x, best_val, best_aspect
        """
        # We need to optimize x (63 vars) and aspect (1 var) together?
        # Or fix aspect and optimize x, then loop?
        # Coupling aspect and x is better for global search but NM is slow in 64D.
        # Let's try optimizing x for a fixed aspect, then do a coarse search on aspect.
        # Actually, let's just include aspect in the vector for the first stage?
        # NM in 64D is risky. Let's stick to 63D for x, and manually sweep aspect.
        
        # However, the current program's weakness is getting stuck in local minima of x for a fixed aspect.
        # Let's use NM on x only.
        
        # Constraints on x? NM doesn't handle bounds well.
        # We rely on the penalty to push x back into bounds.
        
        try:
            res = minimize(objective, x0, args=(aspect,), method='Nelder-Mead', 
                           options={'maxiter': max_iter, 'xatol': 1e-4, 'fatol': 1e-4, 'adaptive': True})
            return res.x, res.fun, res.success
        except Exception:
            return x0, objective(x0, aspect), False

    best_overall_score = -np.inf
    best_overall_result = None
    
    # Define a set of initial aspects to try
    # Perimeter 4 => w+h=2.
    # Square: w=h=1 (aspect=1)
    # Rect: w=1.5, h=0.5 (aspect=3), w=0.5, h=1.5 (aspect=1/3)
    aspects_to_try = [0.5, 0.8, 1.0, 1.25, 1.5, 2.0, 3.0]
    
    # Randomize aspects slightly for diversity in multi-start
    random.seed(42)
    np.random.seed(42)
    
    def get_initial_guess(aspect):
        """
        Generate a hexagonal-ish initial guess that fits inside the rectangle.
        """
        w = 2.0 * aspect / (1.0 + aspect)
        h = 2.0 / (1.0 + aspect)
        
        # Estimate radius based on area
        # Area = w*h. 21 circles.
        # r_approx = sqrt(w*h / (21 * pi)) * 0.9
        r_approx = np.sqrt(w * h / (N * np.pi)) * 0.85
        
        # Create grid
        # Rows and cols to cover N
        # Hex packing:
        # Row 0: r, 3r, 5r...
        # Row 1: 2r, 4r... shifted by r in y (sqrt(3)/2 r)
        
        # Simple grid first, then perturb
        side = int(np.ceil(np.sqrt(N)))
        x0 = np.zeros(3 * N)
        
        for i in range(N):
            row = i // side
            col = i % side
            
            # Center in grid
            cx = (col + 0.5) * (w / side)
            cy = (row + 0.5) * (h / side)
            
            # Add small random jitter to break symmetry
            cx += np.random.uniform(-0.05, 0.05) * (w/side)
            cy += np.random.uniform(-0.05, 0.05) * (h/side)
            
            # Clamp to bounds roughly
            cx = np.clip(cx, r_approx, w - r_approx)
            cy = np.clip(cy, r_approx, h - r_approx)
            
            x0[i*3] = cx
            x0[i*3+1] = cy
            x0[i*3+2] = r_approx
            
        return x0

    # Multi-start loop
    num_starts = 50
    # We have ~130s.
    # Each NM run takes maybe 0.5-2s.
    # 50 starts * 1 aspect = 50 runs.
    # If we do 2 aspects per start, 100 runs.
    
    # Strategy: 
    # 1. Coarse sweep of aspects.
    # 2. For each aspect, do 1-2 NM runs from different seeds.
    
    # To maximize chance, we will iterate through aspects and perform multiple restarts if time permits.
    
    # List of (aspect, seed_idx)
    tasks = []
    for asp in aspects_to_try:
        # 3 restarts per aspect
        for k in range(3):
            tasks.append((asp, k))
    
    random.shuffle(tasks)
    
    for asp, seed_idx in tasks:
        if time.time() > deadline:
            break
            
        x0 = get_initial_guess(asp)
        
        # Add noise to x0 based on seed_idx
        if seed_idx > 0:
            noise_scale = 0.1 * (np.sqrt(2.0 * asp / (1.0 + asp)))
            x0[0::3] += np.random.normal(0, noise_scale, N)
            x0[1::3] += np.random.normal(0, noise_scale, N)
            x0[2::3] *= (1.0 + np.random.normal(0, 0.1, N))
        
        # Run NM
        # Limit iterations to ensure we finish in time
        max_iter = 2000
        x_opt, val, success = run_nelder_mead(x0, asp, max_iter)
        
        # Evaluate validity and score of the result
        # val = penalty - sum_r. 
        # If penalty is small, val is approx -sum_r.
        # We want high sum_r. So low val is good.
        
        # Check if valid
        w = 2.0 * asp / (1.0 + asp)
        h = 2.0 / (1.0 + asp)
        xy = x_opt[0::3]
        yv = x_opt[1::3]
        r = x_opt[2::3]
        
        # Hard check
        valid = True
        if np.any(xy < r - 1e-5) or np.any(xy > w - r + 1e-5) or \
           np.any(yv < r - 1e-5) or np.any(yv > h - r + 1e-5):
            valid = False
            
        if valid:
            dx = xy[pairs[:, 0]] - xy[pairs[:, 1]]
            dy = yv[pairs[:, 0]] - yv[pairs[:, 1]]
            dist = np.sqrt(dx*dx + dy*dy)
            r_sum = r[pairs[:, 0]] + r[pairs[:, 1]]
            if np.any(dist < r_sum - 1e-5):
                valid = False
        
        if valid:
            score = np.sum(r)
            if score > best_overall_score:
                best_overall_score = score
                # Construct result array
                res_arr = np.column_stack((xy, yv, r))
                best_overall_result = res_arr.copy()
        
        # If invalid, we might still want to keep the best "almost valid" for future refinement?
        # For now, we only update best if valid.
        
    if best_overall_result is None:
        # Fallback: Return a valid trivial packing (small circles)
        # Grid of tiny circles
        side = 5 # 25 spots, use 21
        w = 1.0
        h = 1.0
        res = np.zeros((N, 3))
        r = 0.08
        for i in range(N):
            row = i // 5
            col = i % 5
            res[i, 0] = (col + 0.5) * 0.2
            res[i, 1] = (row + 0.5) * 0.2
            res[i, 2] = r
        return res

    # Final Polish: Try to increase radii slightly using a local search on the best result
    # This helps if NM stopped just short of the boundary due to tolerance.
    best_res = best_overall_result
    asp_best = None # We need to track which aspect was used. 
    # Wait, I didn't store asp_best. I should store it.
    # Let's re-derive asp from the bounding box of the result? 
    # No, the result is in absolute coordinates. 
    # The evaluator expects the circles to be in a rectangle of perimeter 4.
    # My `construct` returns circles in absolute coords. 
    # The evaluator will check if they fit in SOME rectangle of perimeter 4?
    # "bounding rectangle has perimeter at most 4"
    # Usually this means the evaluator finds the min bounding box of the circles and checks its perimeter.
    # OR it assumes the rectangle is defined by the circles' extents?
    # "construct() -> array of shape (n, 3): n disjoint circles whose bounding rectangle has perimeter at most 4"
    # This implies the evaluator calculates the bounding box of the returned points and checks if 2*(w+h) <= 4.
    # So I don't need to return the aspect. I just need to ensure the points fit in a 2*(w+h)<=4 box.
    # My optimization assumed a specific w,h. If the points fit in w,h, they fit in the bounding box.
    # However, if the points don't touch the boundaries, the bounding box might be smaller, which is fine.
    # If they touch, the bounding box is w,h.
    
    # So, I just return the coordinates.
    
    # Polish: 
    # Try to expand radii slightly.
    # Since we don't know the "intended" rectangle, we assume the tightest one defined by current points.
    # Actually, it's safer to just return the best valid result found.
    # The NM penalty approach usually drives circles to the boundaries to maximize r.
    
    return best_overall_result
# EVOLVE-BLOCK-END
