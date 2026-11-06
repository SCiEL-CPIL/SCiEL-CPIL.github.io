import numpy as np
import time
import scipy.optimize as opt

N = 200
DEADLINE = 140.0  # Safety margin before 150s kill

def _compute_radius(points):
    """
    Compute the maximum radius for a given set of points.
    r = min(min_pairwise_dist, min_wall_dist) / 2
    """
    # Pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    np.fill_diagonal(dist_sq, np.inf)
    min_center_dist = np.sqrt(np.min(dist_sq))
    
    # Wall distances
    dx = 0.5 - np.abs(points[:, 0])
    dy = 0.5 - np.abs(points[:, 1])
    min_wall_dist = np.min(np.concatenate([dx, dy]))
    
    r = min(min_center_dist, min_wall_dist) / 2.0
    return r

def _generate_hex_grid(n):
    """
    Generate a hexagonal grid of points inside [-0.5, 0.5]^2.
    Returns an array of shape (n, 2).
    """
    # Estimate spacing for hex packing
    # Area per point ~ 1/n. Hex cell area ~ s^2 * sqrt(3)/2
    # s^2 * 0.866 ~ 1/200 => s^2 ~ 0.00577 => s ~ 0.076
    s = np.sqrt(2.0 / (n * np.sqrt(3)))
    
    points = []
    # Generate grid covering a bit more than the square
    n_rows = int(np.ceil(1.0 / (s * np.sqrt(3) / 2))) + 2
    n_cols = int(np.ceil(1.0 / s)) + 2
    
    for r in range(-n_rows, n_rows + 1):
        y = r * s * np.sqrt(3) / 2
        if y < -0.55 or y > 0.55:
            continue
        x_offset = (r % 2) * s / 2.0
        for c in range(-n_cols, n_cols + 1):
            x = c * s + x_offset
            if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                points.append([x, y])
                
    points = np.array(points)
    
    if len(points) > n:
        # Randomly sample n points from the generated grid
        indices = np.random.choice(len(points), n, replace=False)
        points = points[indices]
    elif len(points) < n:
        # If we don't have enough, add random points
        missing = n - len(points)
        random_pts = np.random.uniform(-0.45, 0.45, (missing, 2))
        points = np.vstack([points, random_pts])
        
    return points

def construct():
    start_time = time.time()
    
    # Initial configuration
    np.random.seed(42) # Fixed seed for reproducibility in initialization, though restarts will vary
    points = _generate_hex_grid(N)
    # Small noise to break symmetry
    points += np.random.normal(0, 0.001, points.shape)
    points = np.clip(points, -0.49, 0.49)
    
    current_r = _compute_radius(points)
    best_points = points.copy()
    best_r = current_r
    
    # Objective function for L-BFGS-B
    # Minimizes sum of squared violations for a target radius r_target
    def objective_and_grad(p_flat, r_target):
        p = p_flat.reshape(N, 2)
        E = 0.0
        grad = np.zeros_like(p)
        
        # 1. Boundary violations
        # Distance to each wall
        d_left = p[:, 0] + 0.5
        d_right = 0.5 - p[:, 0]
        d_bottom = p[:, 1] + 0.5
        d_top = 0.5 - p[:, 1]
        
        # Minimum distance to any wall for each point
        # We use a soft min or just check the min component
        # For gradient calculation, we need to know which wall is the closest
        # d_min_i = min(d_left, d_right, d_bottom, d_top)
        
        d_min = np.minimum(np.minimum(d_left, d_right), np.minimum(d_bottom, d_top))
        
        # Violation: r_target - d_min
        viol_b = np.maximum(0.0, r_target - d_min)
        E += np.sum(viol_b**2)
        
        # Gradient for boundary
        # dE/dp = -2 * viol_b * grad(d_min)
        # grad(d_min) is a unit vector pointing towards the closest wall (inward)
        # We compute this per point
        for i in range(N):
            if viol_b[i] > 0:
                x, y = p[i]
                dl = x + 0.5
                dr = 0.5 - x
                db = y + 0.5
                dt = 0.5 - y
                
                # Find the minimum
                min_d = min(dl, dr, db, dt)
                
                gx = 0.0
                gy = 0.0
                
                # If multiple walls are equally close, sum gradients? 
                # Usually not an issue in generic positions.
                if min_d == dl:
                    gx = 1.0
                elif min_d == dr:
                    gx = -1.0
                elif min_d == db:
                    gy = 1.0
                elif min_d == dt:
                    gy = -1.0
                
                # dE/dp_i = -2 * viol_b[i] * [gx, gy]
                # We are minimizing E, so optimizer moves in -grad direction
                # -grad = 2 * viol_b[i] * [gx, gy] -> Points inward. Correct.
                grad[i, 0] += -2.0 * viol_b[i] * gx
                grad[i, 1] += -2.0 * viol_b[i] * gy

        # 2. Pairwise violations
        diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=2)
        np.fill_diagonal(dist_sq, np.inf)
        dist = np.sqrt(dist_sq)
        
        # Violation: 2*r_target - dist
        viol_pairs = np.maximum(0.0, 2.0 * r_target - dist)
        
        # Sum of squares (upper triangle to avoid double counting in E, but gradient needs care)
        # E += 0.5 * sum_{i!=j} viol_{ij}^2
        # Using upper triangle indices
        i_idx, j_idx = np.triu_indices(N, k=1)
        v = viol_pairs[i_idx, j_idx]
        d = dist[i_idx, j_idx]
        
        E += np.sum(v**2)
        
        # Gradient
        # For pair (i, j):
        # T = (2r - d)^2
        # dT/dp_i = -2(2r-d) * (p_i - p_j)/d
        # dT/dp_j = -2(2r-d) * (p_j - p_i)/d = 2(2r-d) * (p_i - p_j)/d
        
        mask = v > 0
        if np.any(mask):
            i_m = i_idx[mask]
            j_m = j_idx[mask]
            v_m = v[mask]
            d_m = d[mask]
            
            # Avoid division by zero
            d_safe = np.where(d_m > 1e-10, d_m, 1e-10)
            
            vec = p[i_m] - p[j_m]
            unit = vec / d_safe[:, np.newaxis]
            
            # factor = 2 * v_m / d_safe
            # grad[i] += -2 * v_m * unit
            # grad[j] +=  2 * v_m * unit
            
            np.add.at(grad, i_m, -2.0 * v_m[:, np.newaxis] * unit)
            np.add.at(grad, j_m,  2.0 * v_m[:, np.newaxis] * unit)
            
        return E, grad.flatten()

    bounds = [(-0.5, 0.5)] * (2 * N)
    
    # Optimization Loop
    # We will try to increase the radius step by step.
    r_target = current_r
    step_r = 0.0005  # Initial step size
    max_iter_per_step = 50  # L-BFGS-B iterations per radius step
    
    # Main loop: increase radius
    while time.time() - start_time < DEADLINE:
        # Try to increase radius
        r_target_new = r_target + step_r
        
        # If we are very close to the limit, use smaller steps or stop
        remaining_time = DEADLINE - (time.time() - start_time)
        if remaining_time < 5.0:
            break
            
        # Optimize for the new target radius
        p_flat = points.flatten()
        
        try:
            res = opt.minimize(
                objective_and_grad, 
                p_flat, 
                args=(r_target_new,), 
                jac=True, 
                method='L-BFGS-B', 
                bounds=bounds, 
                options={'maxiter': max_iter_per_step, 'gtol': 1e-6, 'ftol': 1e-8}
            )
            
            new_points = res.x.reshape(N, 2)
            # Check if the new configuration is actually valid for r_target_new
            # Calculate actual radius
            actual_r = _compute_radius(new_points)
            
            # If the actual radius is at least r_target_new (or close to it), we succeeded
            # We accept the new points if they are better than current best
            if actual_r > best_r:
                best_r = actual_r
                best_points = new_points.copy()
                
                # Update current points to this new best
                points = new_points.copy()
                r_target = actual_r # Set target to current achieved radius
                
                # If we improved significantly, we can try a slightly larger step next time?
                # Or keep it steady. Steady is safer.
                
            else:
                # If we didn't improve, maybe the step was too big.
                # Revert to previous points? Or just keep trying with same r_target?
                # If we keep trying with same r_target, we might converge.
                # But if we failed to reach r_target_new, we might be stuck.
                # Let's revert points to the last known good state (which is best_points)
                # and try a smaller step or just break if no progress in many steps.
                
                # For now, revert to best_points to ensure we don't degrade
                points = best_points.copy()
                r_target = best_r
                
                # Reduce step size if we fail?
                # step_r *= 0.9 
                
                # To avoid infinite loop of failure, if we fail, we can try to
                # "relax" the current configuration by optimizing for the CURRENT r_target
                # to see if we can find a better local minimum before increasing.
                
                # Let's do a few iterations for the current r_target
                res2 = opt.minimize(
                    objective_and_grad, 
                    points.flatten(), 
                    args=(r_target,), 
                    jac=True, 
                    method='L-BFGS-B', 
                    bounds=bounds, 
                    options={'maxiter': max_iter_per_step, 'gtol': 1e-6, 'ftol': 1e-8}
                )
                relaxed_points = res2.x.reshape(N, 2)
                relaxed_r = _compute_radius(relaxed_points)
                
                if relaxed_r > best_r:
                    best_r = relaxed_r
                    best_points = relaxed_points.copy()
                    points = relaxed_points.copy()
                    r_target = best_r
                else:
                    # No improvement. Break to save time or try different init?
                    # Since we are in a loop, let's just break out of the radius increase loop
                    # and rely on the best found so far.
                    # Or, we can try a random restart if we have time.
                    break
                    
        except Exception as e:
            # If optimization fails, break
            break
            
    return best_points

# Entry point required by the task
def construct():
    # The function above is already named construct.
    # Just return it.
    return _construct_impl()

# To keep the signature exactly as requested and avoid name conflicts in definition,
# let's rename the internal one and have a wrapper.
# Wait, the prompt says "Keep the function names and signatures of the seed program."
# The seed program had `def construct():`.
# I defined `def construct():` twice in the draft above. I will fix that in the final output.

# Final Structure:
# 1. Helper functions
# 2. def construct(): ...

# Let's re-write cleanly.

import numpy as np
import time
import scipy.optimize as opt

N = 200
DEADLINE = 140.0

def _compute_radius(points):
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    np.fill_diagonal(dist_sq, np.inf)
    min_center_dist = np.sqrt(np.min(dist_sq))
    dx = 0.5 - np.abs(points[:, 0])
    dy = 0.5 - np.abs(points[:, 1])
    min_wall_dist = np.min(np.concatenate([dx, dy]))
    r = min(min_center_dist, min_wall_dist) / 2.0
    return r

def _generate_hex_grid(n):
    s = np.sqrt(2.0 / (n * np.sqrt(3)))
    points = []
    n_rows = int(np.ceil(1.0 / (s * np.sqrt(3) / 2))) + 2
    n_cols = int(np.ceil(1.0 / s)) + 2
    for r in range(-n_rows, n_rows + 1):
        y = r * s * np.sqrt(3) / 2
        if y < -0.55 or y > 0.55:
            continue
        x_offset = (r % 2) * s / 2.0
        for c in range(-n_cols, n_cols + 1):
            x = c * s + x_offset
            if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                points.append([x, y])
    points = np.array(points)
    if len(points) > n:
        indices = np.random.choice(len(points), n, replace=False)
        points = points[indices]
    elif len(points) < n:
        missing = n - len(points)
        random_pts = np.random.uniform(-0.45, 0.45, (missing, 2))
        points = np.vstack([points, random_pts])
    return points

def construct():
    start_time = time.time()
    
    np.random.seed(42)
    points = _generate_hex_grid(N)
    points += np.random.normal(0, 0.001, points.shape)
    points = np.clip(points, -0.49, 0.49)
    
    current_r = _compute_radius(points)
    best_points = points.copy()
    best_r = current_r
    
    def objective_and_grad(p_flat, r_target):
        p = p_flat.reshape(N, 2)
        E = 0.0
        grad = np.zeros_like(p)
        
        d_left = p[:, 0] + 0.5
        d_right = 0.5 - p[:, 0]
        d_bottom = p[:, 1] + 0.5
        d_top = 0.5 - p[:, 1]
        
        d_min = np.minimum(np.minimum(d_left, d_right), np.minimum(d_bottom, d_top))
        viol_b = np.maximum(0.0, r_target - d_min)
        E += np.sum(viol_b**2)
        
        for i in range(N):
            if viol_b[i] > 0:
                x, y = p[i]
                dl = x + 0.5
                dr = 0.5 - x
                db = y + 0.5
                dt = 0.5 - y
                min_d = min(dl, dr, db, dt)
                gx = 0.0
                gy = 0.0
                if min_d == dl:
                    gx = 1.0
                elif min_d == dr:
                    gx = -1.0
                elif min_d == db:
                    gy = 1.0
                elif min_d == dt:
                    gy = -1.0
                grad[i, 0] += -2.0 * viol_b[i] * gx
                grad[i, 1] += -2.0 * viol_b[i] * gy

        diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=2)
        np.fill_diagonal(dist_sq, np.inf)
        dist = np.sqrt(dist_sq)
        
        viol_pairs = np.maximum(0.0, 2.0 * r_target - dist)
        i_idx, j_idx = np.triu_indices(N, k=1)
        v = viol_pairs[i_idx, j_idx]
        d = dist[i_idx, j_idx]
        
        E += np.sum(v**2)
        
        mask = v > 0
        if np.any(mask):
            i_m = i_idx[mask]
            j_m = j_idx[mask]
            v_m = v[mask]
            d_m = d[mask]
            d_safe = np.where(d_m > 1e-10, d_m, 1e-10)
            vec = p[i_m] - p[j_m]
            unit = vec / d_safe[:, np.newaxis]
            np.add.at(grad, i_m, -2.0 * v_m[:, np.newaxis] * unit)
            np.add.at(grad, j_m,  2.0 * v_m[:, np.newaxis] * unit)
            
        return E, grad.flatten()

    bounds = [(-0.5, 0.5)] * (2 * N)
    r_target = current_r
    step_r = 0.0005
    max_iter_per_step = 50
    
    while time.time() - start_time < DEADLINE:
        remaining_time = DEADLINE - (time.time() - start_time)
        if remaining_time < 5.0:
            break
            
        r_target_new = r_target + step_r
        
        try:
            res = opt.minimize(
                objective_and_grad, 
                points.flatten(), 
                args=(r_target_new,), 
                jac=True, 
                method='L-BFGS-B', 
                bounds=bounds, 
                options={'maxiter': max_iter_per_step, 'gtol': 1e-6, 'ftol': 1e-8}
            )
            
            new_points = res.x.reshape(N, 2)
            actual_r = _compute_radius(new_points)
            
            if actual_r > best_r:
                best_r = actual_r
                best_points = new_points.copy()
                points = new_points.copy()
                r_target = actual_r
            else:
                # Relax current configuration
                res2 = opt.minimize(
                    objective_and_grad, 
                    points.flatten(), 
                    args=(r_target,), 
                    jac=True, 
                    method='L-BFGS-B', 
                    bounds=bounds, 
                    options={'maxiter': max_iter_per_step, 'gtol': 1e-6, 'ftol': 1e-8}
                )
                relaxed_points = res2.x.reshape(N, 2)
                relaxed_r = _compute_radius(relaxed_points)
                
                if relaxed_r > best_r:
                    best_r = relaxed_r
                    best_points = relaxed_points.copy()
                    points = relaxed_points.copy()
                    r_target = best_r
                else:
                    break
                    
        except Exception:
            break
            
    return best_points
