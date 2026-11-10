import numpy as np
import time
from scipy.optimize import minimize

N = 50
DEADLINE = 115.0  # Safety margin from 120s

def compute_radius(positions):
    """
    Compute the maximum radius for the given positions.
    Returns (radius, is_valid).
    """
    max_abs = np.max(np.abs(positions), axis=1)
    bound_dists = 0.5 - max_abs
    
    if np.any(bound_dists < -1e-9):
        return -1.0, False
        
    min_bound = np.min(bound_dists)
    
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=-1)
    np.fill_diagonal(dists_sq, np.inf)
    
    min_dist_sq = np.min(dists_sq)
    min_dist = np.sqrt(min_dist_sq)
    
    r = min(min_bound, min_dist / 2.0)
    
    return r, True

def penalty_function(flat_positions, R):
    """
    Calculate the penalty for a configuration given a target radius R.
    Penalty is the sum of squared overlaps with other circles and boundaries.
    """
    positions = flat_positions.reshape(N, 2)
    
    # Boundary penalty
    max_abs = np.max(np.abs(positions), axis=1)
    bound_dists = 0.5 - max_abs
    # We want bound_dists >= R
    # Penalty: max(0, R - bound_dists)^2
    boundary_violations = np.maximum(0, R - bound_dists)
    penalty_bound = np.sum(boundary_violations**2)
    
    # Pairwise penalty
    # We need min distance >= 2R
    # Penalty: max(0, 2R - d_ij)^2
    # Vectorized calculation of all pairwise distances is O(N^2), which is 2500 ops.
    # This is called many times by the optimizer.
    # We can optimize this by only computing upper triangle? 
    # For simplicity and speed in Python, let's try to be efficient.
    
    # Calculate all pairwise squared distances
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=-1)
    np.fill_diagonal(dists_sq, np.inf)
    
    # We need d_ij >= 2R
    # d_ij = sqrt(dists_sq)
    # Condition: sqrt(dists_sq) < 2R  => dists_sq < 4R^2
    # Violation: (2R - sqrt(dists_sq))^2
    # To avoid sqrt, we can work with squared distances?
    # Let x = dists_sq. We want x >= (2R)^2 = 4R^2.
    # If x < 4R^2, violation is (2R - sqrt(x))^2 = 4R^2 - 4R*sqrt(x) + x.
    # This still has sqrt.
    
    # Let's just use the direct formula, it's N=50 so 1225 pairs.
    # 1225 sqrts is fast in numpy.
    
    dists = np.sqrt(dists_sq)
    # Mask upper triangle to count each pair once
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    pair_dists = dists[mask]
    
    # Violations where pair_dists < 2R
    violations = np.maximum(0, 2*R - pair_dists)
    penalty_pairs = np.sum(violations**2)
    
    return penalty_bound + penalty_pairs

def construct():
    start_time = time.time()
    
    best_radius = 0.0
    best_positions = None
    
    # Known good radius is ~0.071377
    # We will try to optimize for a target radius slightly higher than current best
    # and see if we can find a valid config.
    
    # Initializations
    inits = []
    
    # 1. Hexagonal Grid
    s = 1.0
    points = []
    for j in range(-15, 16):
        for i in range(-15, 16):
            x = i * s + (j % 2) * s / 2.0
            y = j * s * np.sqrt(3.0) / 2.0
            points.append([x, y])
    points = np.array(points)
    dists_to_center = np.sum(points**2, axis=1)
    indices = np.argsort(dists_to_center)[:N]
    hex_pos = points[indices].copy()
    max_abs = np.max(np.abs(hex_pos))
    if max_abs > 0:
        hex_pos = hex_pos * (0.5 / max_abs)
    # Add small noise
    hex_pos += np.random.normal(0, 0.01, size=hex_pos.shape)
    hex_pos = np.clip(hex_pos, -0.4999, 0.4999)
    inits.append(hex_pos)
    
    # 2. Random
    rand_pos = np.random.uniform(-0.5, 0.5, size=(N, 2))
    inits.append(rand_pos)
    
    # 3. Grid
    grid_pos = np.random.uniform(-0.45, 0.45, size=(N, 2))
    inits.append(grid_pos)
    
    # We will use Powell optimization to minimize the penalty for a fixed R.
    # We start with a target R and if we find a solution with penalty ~0, 
    # we can try to increase R.
    
    # Let's iterate over several target radii, from low to high.
    # Or, better: for each init, run optimization to minimize penalty for R_target.
    # If successful (penalty < 1e-6), record the positions and calculate actual radius.
    
    # To maximize performance, we can just run the optimizer to minimize the penalty 
    # for a R that is slightly above the current best known.
    
    # Let's use a binary search style or incremental approach.
    # Start with R = 0.07
    R_target = 0.070
    
    # We have ~110 seconds.
    # Powell is slow. Let's limit the number of function evaluations.
    
    for init_pos in inits:
        if time.time() - start_time > DEADLINE:
            break
            
        # Flatten for scipy
        x0 = init_pos.flatten()
        
        # Define objective
        def obj(x):
            return penalty_function(x, R_target)
        
        # Run Powell
        # maxiter is crucial
        # We don't know how many iters we can afford.
        # Let's set a timeout based on remaining time?
        # Powell doesn't support timeout directly.
        # We can wrap the function to check time, but it might be messy.
        # Let's just run it with a reasonable maxiter and see.
        
        try:
            res = minimize(obj, x0, method='Powell', options={'maxiter': 500, 'maxfev': 2000})
        except Exception:
            continue
            
        if time.time() - start_time > DEADLINE:
            break
            
        new_pos = res.x.reshape(N, 2)
        final_penalty = obj(new_pos)
        
        # Check if it's a valid configuration for R_target
        if final_penalty < 1e-4:
            # Calculate actual radius
            r, valid = compute_radius(new_pos)
            if valid and r > best_radius:
                best_radius = r
                best_positions = new_pos.copy()
                
                # If we found a good one, we can try to push R higher
                # But we are in a loop over inits. 
                # Let's try to do a second pass with higher R if we have time.
                
    # Second phase: Try to improve the best found radius
    if best_positions is not None:
        # Try to optimize for a higher radius
        # Use the best positions as start
        R_target2 = best_radius * 1.01
        x0 = best_positions.flatten()
        
        def obj2(x):
            return penalty_function(x, R_target2)
            
        try:
            # Use more iterations if time permits
            remaining = DEADLINE - (time.time() - start_time)
            maxiter = int(remaining * 10) # Rough estimate
            if maxiter < 100:
                maxiter = 100
            res = minimize(obj2, x0, method='Powell', options={'maxiter': maxiter, 'maxfev': maxiter*4})
            if time.time() - start_time > DEADLINE:
                return best_positions
            
            new_pos = res.x.reshape(N, 2)
            final_penalty = obj2(new_pos)
            
            if final_penalty < 1e-4:
                r, valid = compute_radius(new_pos)
                if valid and r > best_radius:
                    best_radius = r
                    best_positions = new_pos.copy()
                    
        except Exception:
            pass

    # If no valid position found (shouldn't happen), return hex grid
    if best_positions is None:
        best_positions = inits[0]
        
    return best_positions
