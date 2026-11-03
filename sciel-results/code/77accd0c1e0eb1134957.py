"""Circle packing in perimeter-4 rectangle – Global Differential Evolution."""
import numpy as np
import time
from scipy.optimize import differential_evolution

N = 21

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s kill

    # Variables:
    # x_coords (N), y_coords (N), radius (1), width (1)
    # Height is determined by width: h = 2 - w
    # Total dimensions: 2N + 2 = 44
    
    def objective(params):
        # Unpack
        xs = params[:N]
        ys = params[N:2*N]
        r = params[2*N]
        w = params[2*N+1]
        h = 2.0 - w
        
        # Check bounds for r and w
        # r must be positive
        if r <= 0:
            return 1e6
        # w must be such that h > 0 and w > 0
        if w <= 0 or h <= 0:
            return 1e6
        
        # Wall constraints
        # min distance to walls must be >= r
        min_x = np.min(xs)
        max_x = np.max(xs)
        min_y = np.min(ys)
        max_y = np.max(ys)
        
        pen = 0.0
        # If a circle goes outside, penalize heavily
        # We want to maximize r, so minimize -r + penalty
        # Penalty for wall violation
        wall_violations = 0
        if min_x < r:
            wall_violations += (r - min_x)
        if max_x > w - r:
            wall_violations += (max_x - (w - r))
        if min_y < r:
            wall_violations += (r - min_y)
        if max_y > h - r:
            wall_violations += (max_y - (h - r))
            
        if wall_violations > 0:
            pen += 1000.0 * wall_violations
            
        # Pairwise constraints
        # dist(i,j) >= 2r
        # We can vectorize distance calculation
        # Create matrix of coordinates
        coords = np.vstack((xs, ys)).T
        # Calculate pairwise squared distances
        # diff[i,j] = coords[i] - coords[j]
        # To save memory/time for N=21, loop is fine, but vectorized is faster
        # Using broadcasting
        diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=2)
        
        # We only care about upper triangle
        # dist >= 2r  => dist_sq >= 4r^2
        # Violation if dist_sq < 4r^2
        # Penalty sum of (2r - dist) for all pairs where dist < 2r
        
        # Avoid sqrt for pairs that are definitely far? 
        # For N=21, 210 pairs. Sqrt is cheap.
        dist = np.sqrt(dist_sq)
        
        # Mask for upper triangle
        mask = np.triu(np.ones((N, N), dtype=bool), k=1)
        dists = dist[mask]
        
        r_sum = 2.0 * r
        violations = np.maximum(0, r_sum - dists)
        if len(violations) > 0:
            pen += 1000.0 * np.sum(violations)
            
        return -r + pen

    # Bounds for differential evolution
    # xs, ys: [0, 2] (loose bounds, perimeter 4 implies max dim 2 if other is 0, but practically < 2)
    # r: [0.01, 0.5] (max possible r is roughly 1/6 if 6x1 grid? 21 circles in 2x1 area. Area 2. 21 * pi * r^2 <= 2 => r^2 <= 2/(21*pi) ~ 0.03 => r ~ 0.17. But packing density < 1. So 0.2 is a safe upper bound for single r. Let's go up to 0.3 to be safe.)
    # w: [0.1, 1.9] (h = 2-w, so w cannot be 0 or 2)
    
    bounds = []
    for _ in range(N):
        bounds.append((0.0, 2.0)) # x
    for _ in range(N):
        bounds.append((0.0, 2.0)) # y
    bounds.append((0.01, 0.3)) # r
    bounds.append((0.2, 1.8)) # w

    # Initial population: 
    # DE can be slow. Let's give it a decent initial population if possible, or just let it run.
    # Since DE is global, it handles non-convexity well.
    
    # We will run DE with a time limit.
    # DE in scipy does not support time limits directly, only maxiter.
    # We estimate maxiter.
    # Population size = 15 * len(bounds) = 15 * 44 = 660.
    # Each function eval is fast (N=21).
    # Let's try maxiter=200. 
    # 660 * 200 = 132,000 evaluations.
    # If each eval is ~10us, that's 1.3s. 
    # If 1ms, that's 132s. 
    # Let's optimize the objective function to be fast.
    
    # Optimized objective
    def fast_objective(params):
        xs = params[:N]
        ys = params[N:2*N]
        r = params[2*N]
        w = params[2*N+1]
        h = 2.0 - w
        
        if r <= 0 or w <= 0 or h <= 0:
            return 1e6
            
        # Wall checks
        min_x = xs.min()
        max_x = xs.max()
        min_y = ys.min()
        max_y = ys.max()
        
        pen = 0.0
        if min_x < r: pen += 1000.0 * (r - min_x)
        if max_x > w - r: pen += 1000.0 * (max_x - (w - r))
        if min_y < r: pen += 1000.0 * (r - min_y)
        if max_y > h - r: pen += 1000.0 * (max_y - (h - r))
        
        if pen > 1e-5:
            # Early exit if wall violations are huge
            pass # Still need to check overlaps? No, if wall violation is huge, -r + pen is dominated by pen. 
                 # But we want to guide towards feasibility. 
                 # Actually, if we just return -r + pen, it's fine.
                 
        # Pairwise
        # Use numpy vectorization
        # coords = np.column_stack((xs, ys))
        # dist_sq = ...
        
        # Alternative: loop might be faster for N=21 due to overhead of creating arrays?
        # Let's try a simple loop with local variables for speed
        min_dist = np.inf
        # Actually, we need sum of violations, not just min.
        # Let's stick to vectorized, it's usually fast enough for N=21.
        
        # Create coordinate matrix
        # To avoid vstack overhead, we can do:
        # dx = xs[:, None] - xs[None, :]
        # dy = ys[:, None] - ys[None, :]
        # dist_sq = dx*dx + dy*dy
        # This creates 21x21 arrays. Very fast.
        
        dx = xs[:, None] - xs[None, :]
        dy = ys[:, None] - ys[None, :]
        dist_sq = dx*dx + dy*dy
        
        # Upper triangle indices
        i_upper, j_upper = np.triu_indices(N, k=1)
        dists = np.sqrt(dist_sq[i_upper, j_upper])
        
        r_sum = 2.0 * r
        violations = r_sum - dists
        # Only positive violations contribute
        # np.sum(np.where(violations > 0, violations, 0))
        # Or np.sum(violations[violations > 0])
        pos_viol = violations[violations > 0]
        if len(pos_viol) > 0:
            pen += 1000.0 * np.sum(pos_viol)
            
        return -r + pen

    # Run DE
    # seed=42 for reproducibility
    # maxiter: let's start with 50 and increase if time permits?
    # DE is stochastic. 
    # Let's run one instance.
    
    # To be safe with time, we can wrap the DE call in a check?
    # SciPy DE doesn't support callbacks for stopping.
    # We must estimate.
    # N_params = 44.
    # Pop size = 15 * 44 = 660.
    # Iter 1: 660 evals.
    # Iter k: 660 evals.
    # Total evals = 660 * maxiter.
    
    # Let's profile fast_objective roughly.
    # It's ~20 numpy ops on small arrays. Should be < 50us.
    # 10,000 evals ~ 0.5s.
    # So 100 iterations ~ 66,000 evals ~ 3s.
    # We can afford maxiter=200.
    
    try:
        res = differential_evolution(
            fast_objective, 
            bounds, 
            seed=42, 
            maxiter=200, 
            tol=1e-8, 
            polish=True  # Use L-BFGS-B at the end to refine
        )
        
        if res.success or res.fun < 0: # If it found a valid solution (fun = -r + pen, so negative means r > pen)
            # Extract best
            best_params = res.x
            best_val = -res.fun # This is r - pen/1000? No, res.fun is the objective value.
            # Objective is -r + pen. So -res.fun is r - pen.
            # We want the actual r from the parameters to ensure validity.
            
            r_opt = best_params[2*N]
            w_opt = best_params[2*N+1]
            h_opt = 2.0 - w_opt
            xs_opt = best_params[:N]
            ys_opt = best_params[N:2*N]
            
            # Validate
            # Check walls
            min_x = np.min(xs_opt)
            max_x = np.max(xs_opt)
            min_y = np.min(ys_opt)
            max_y = np.max(ys_opt)
            
            r_wall = min(min_x, w_opt - max_x, min_y, h_opt - max_y)
            
            # Check pairs
            dx = xs_opt[:, None] - xs_opt[None, :]
            dy = ys_opt[:, None] - ys_opt[None, :]
            dist_sq = dx*dx + dy*dy
            i_upper, j_upper = np.triu_indices(N, k=1)
            dists = np.sqrt(dist_sq[i_upper, j_upper])
            r_pair = np.min(dists) / 2.0
            
            r_final = min(r_wall, r_pair)
            
            # If r_final is close to r_opt, it's good.
            # If r_final < r_opt, shrink r_opt to r_final.
            if r_final > 0:
                # Construct result
                circles = np.zeros((N, 3))
                circles[:, 0] = xs_opt
                circles[:, 1] = ys_opt
                circles[:, 2] = r_final * 0.9999 # Slight shrink for safety
                
                # Return
                return circles
            else:
                # Fallback
                return _fallback()
        else:
            # Try to extract best from res even if not "success"
            if hasattr(res, 'x') and res.x is not None:
                best_params = res.x
                r_opt = best_params[2*N]
                w_opt = best_params[2*N+1]
                h_opt = 2.0 - w_opt
                xs_opt = best_params[:N]
                ys_opt = best_params[N:2*N]
                
                min_x = np.min(xs_opt)
                max_x = np.max(xs_opt)
                min_y = np.min(ys_opt)
                max_y = np.max(ys_opt)
                
                r_wall = min(min_x, w_opt - max_x, min_y, h_opt - max_y)
                
                dx = xs_opt[:, None] - xs_opt[None, :]
                dy = ys_opt[:, None] - ys_opt[None, :]
                dist_sq = dx*dx + dy*dy
                i_upper, j_upper = np.triu_indices(N, k=1)
                dists = np.sqrt(dist_sq[i_upper, j_upper])
                r_pair = np.min(dists) / 2.0
                
                r_final = min(r_wall, r_pair)
                
                if r_final > 0.05: # Accept if reasonable
                    circles = np.zeros((N, 3))
                    circles[:, 0] = xs_opt
                    circles[:, 1] = ys_opt
                    circles[:, 2] = r_final * 0.9999
                    return circles
                    
    except Exception as e:
        pass

    return _fallback()

def _fallback():
    # Simple 3x7 grid
    w, h = 1.4, 0.6
    centers = np.zeros((N, 2))
    idx = 0
    for i in range(3):
        for j in range(7):
            centers[idx, 0] = (j + 0.5) * w / 7
            centers[idx, 1] = (i + 0.5) * h / 3
            idx += 1
    
    # Calc r
    min_x = np.min(centers[:, 0])
    max_x = np.max(centers[:, 0])
    min_y = np.min(centers[:, 1])
    max_y = np.max(centers[:, 1])
    r_wall = min(min_x, w - max_x, min_y, h - max_y)
    
    dx = centers[:, 0][:, None] - centers[:, 0][None, :]
    dy = centers[:, 1][:, None] - centers[:, 1][None, :]
    dist_sq = dx*dx + dy*dy
    i_upper, j_upper = np.triu_indices(N, k=1)
    dists = np.sqrt(dist_sq[i_upper, j_upper])
    r_pair = np.min(dists) / 2.0
    
    r_final = min(r_wall, r_pair) * 0.9999
    
    circles = np.zeros((N, 3))
    circles[:, :2] = centers
    circles[:, 2] = r_final
    
    return circles
