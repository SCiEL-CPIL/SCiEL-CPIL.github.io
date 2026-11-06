"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time

# Constants
N_CIRCLES = 200
SQUARE_HALF = 0.5
TIME_LIMIT = 145.0  # Safety margin before 150s kill
START_TIME = time.time()

def _time_remaining():
    return TIME_LIMIT - (time.time() - START_TIME)

def _compute_radius(centers):
    """Compute the maximum valid radius for a given set of centers."""
    if len(centers) != N_CIRCLES:
        return -1.0
    
    # 1. Pairwise distances
    # Using broadcasting for N=200 is fine (200x200 matrix)
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists) / 2.0
    
    # 2. Wall distances
    # Distance to nearest wall for each point
    # x in [-0.5, 0.5], y in [-0.5, 0.5]
    dx = np.abs(centers[:, 0])
    dy = np.abs(centers[:, 1])
    min_dist_to_wall = np.min(SQUARE_HALF - np.stack([dx, dy], axis=1), axis=1)
    global_min_wall = np.min(min_dist_to_wall)
    
    # 3. Final radius
    r = min(min_pair_dist, global_min_wall)
    
    # Check validity: centers must be strictly inside? 
    # "A centre on or outside the square... makes the candidate invalid."
    # So r must be > 0.
    if r <= 0:
        return -1.0
        
    return r

def _init_hex_grid():
    """Initialize points on a hexagonal grid clipped to the square."""
    # Estimate spacing d for ~200 points in area 1.
    # Area per hex point ~ d^2 * sqrt(3) / 2
    # 200 * d^2 * sqrt(3) / 2 ~ 1 => d^2 ~ 2 / (200 * 1.732) ~ 0.00577 => d ~ 0.076
    d = 0.076
    
    points = []
    # y range
    y_max = SQUARE_HALF
    y_min = -SQUARE_HALF
    
    # j index for rows
    # y = j * d * sqrt(3) / 2
    row_height = d * np.sqrt(3) / 2
    j_max = int(np.ceil(y_max / row_height))
    j_min = int(np.floor(y_min / row_height))
    
    for j in range(j_min, j_max + 1):
        y = j * row_height
        if y < y_min or y > y_max:
            continue
        
        # x offset for staggered rows
        x_offset = (j % 2) * d / 2.0
        
        # x range
        x_max = SQUARE_HALF
        x_min = -SQUARE_HALF
        
        i_max = int(np.ceil((x_max - x_offset) / d))
        i_min = int(np.floor((x_min - x_offset) / d))
        
        for i in range(i_min, i_max + 1):
            x = i * d + x_offset
            if x < x_min or x > x_max:
                continue
            points.append([x, y])
            
    points = np.array(points)
    
    # If we have too many points, we need to select N_CIRCLES.
    # Strategy: Keep points that are well-distributed.
    # A simple heuristic: sort by distance to center? No, that clusters.
    # Better: Use a grid-like selection or just take the first N?
    # Taking the first N might be biased if the grid generation is not centered.
    # Let's center the grid first.
    
    if len(points) > 0:
        # Center the grid
        center = np.mean(points, axis=0)
        points = points - center
        
        # Re-clip to square? No, the grid was generated within the square.
        # But shifting might push some out.
        # Let's just filter out points that are outside after shifting.
        mask = (np.abs(points[:, 0]) <= SQUARE_HALF) & (np.abs(points[:, 1]) <= SQUARE_HALF)
        points = points[mask]
        
        # If still too many, we need to drop some.
        # Drop points closest to the boundary? Or random?
        # Dropping boundary points might help if the grid is slightly too dense at edges.
        # But we want to maximize the min distance.
        # Let's just take a random subset if we have too many, or the first N.
        # Actually, a better initialization for local search is to have exactly N points.
        # If len(points) > N_CIRCLES:
        #    We can try to select the most "central" ones?
        #    Or just take the first N_CIRCLES.
        
        if len(points) > N_CIRCLES:
            # Sort by distance to origin?
            dists_origin = np.linalg.norm(points, axis=1)
            indices = np.argsort(dists_origin)
            # Take the N_CIRCLES closest to center?
            # This might create a cluster.
            # Alternative: Take a random subset.
            # Let's try taking the first N_CIRCLES from the sorted list?
            # No, let's just use the first N_CIRCLES from the unsorted list, 
            # assuming the grid generation is roughly symmetric.
            # To be safe, let's shuffle and take N?
            # Shuffling breaks the grid structure which is good for initialization.
            # But a structured start is often better for local search.
            # Let's try to select points that form a regular pattern.
            # For now, just take the first N.
            points = points[:N_CIRCLES]
        elif len(points) < N_CIRCLES:
            # Pad with random points
            remaining = N_CIRCLES - len(points)
            new_points = np.random.uniform(-SQUARE_HALF, SQUARE_HALF, size=(remaining, 2))
            points = np.vstack([points, new_points])
            
    else:
        # Fallback to square grid
        m = int(np.ceil(np.sqrt(N_CIRCLES)))
        x = (np.arange(m) + 0.5) / m - 0.5
        y = (np.arange(m) + 0.5) / m - 0.5
        X, Y = np.meshgrid(x, y)
        points = np.column_stack([X.ravel(), Y.ravel()])[:N_CIRCLES]
        
    return points

def _optimize(pts, time_budget):
    """
    Optimize the configuration using gradient ascent on the soft-min of the radius.
    """
    start_time = time.time()
    deadline = start_time + time_budget
    
    # Soft-min parameter
    # beta controls the sharpness. Higher beta -> closer to true min.
    # Too high beta -> unstable gradients.
    beta = 100.0
    
    # Learning rate
    lr = 0.01
    momentum = 0.9
    
    # Velocity
    v = np.zeros_like(pts)
    
    best_val = -np.inf
    best_pts = pts.copy()
    
    steps = 0
    max_steps = 10000 # Safety cap
    
    while time.time() < deadline and steps < max_steps:
        # Compute distances
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        pair_dists = dists / 2.0
        
        # Wall distances
        wall_dists = SQUARE_HALF - np.abs(pts)
        min_wall_dist_per_point = np.min(wall_dists, axis=1)
        
        # Combine all distances
        # We want to maximize the minimum of all these distances.
        all_dists = np.concatenate([pair_dists.ravel(), min_wall_dist_per_point])
        
        # Soft min calculation
        # To avoid overflow, subtract max
        m = np.max(all_dists)
        exp_vals = np.exp(-beta * (all_dists - m))
        sum_exp = np.sum(exp_vals)
        
        # soft_min = m - (1/beta) * log(sum_exp)
        # Gradient of soft_min wrt all_dists:
        # w_k = exp(-beta * (all_dists_k - m)) / sum_exp
        weights = exp_vals / sum_exp
        
        # Gradient wrt pts
        grad = np.zeros_like(pts)
        
        # 1. Wall gradients
        wall_weights = weights[N_CIRCLES * N_CIRCLES:]
        
        # For each point, the min wall distance is min(0.5-|x|, 0.5-|y|)
        # Gradient is non-zero only for the active coordinate.
        abs_x = np.abs(pts[:, 0])
        abs_y = np.abs(pts[:, 1])
        dist_x = SQUARE_HALF - abs_x
        dist_y = SQUARE_HALF - abs_y
        
        # Masks for which coordinate is the minimum
        mask_x = dist_x < dist_y
        mask_y = dist_y < dist_x
        mask_eq = (dist_x == dist_y)
        
        sign_x = np.sign(pts[:, 0])
        sign_y = np.sign(pts[:, 1])
        
        # d(0.5 - |x|)/dx = -sign(x)
        # d(0.5 - |y|)/dy = -sign(y)
        
        grad[:, 0] += np.where(mask_x, wall_weights, 0) * (-sign_x)
        grad[:, 1] += np.where(mask_y, wall_weights, 0) * (-sign_y)
        
        # Handle equality (split weight)
        grad[:, 0] += np.where(mask_eq, wall_weights * 0.5, 0) * (-sign_x)
        grad[:, 1] += np.where(mask_eq, wall_weights * 0.5, 0) * (-sign_y)
        
        # 2. Pair gradients
        pair_weights = weights[:N_CIRCLES * N_CIRCLES].reshape(N_CIRCLES, N_CIRCLES)
        
        # d(pair_dists_ij)/d(c_i) = -0.5 * (c_i - c_j) / dists_ij
        # coeff[i, j] = -0.5 * pair_weights[i, j] / dists[i, j]
        
        with np.errstate(divide='ignore', invalid='ignore'):
            coeff = -0.5 * pair_weights / dists
        coeff = np.where(np.eye(N_CIRCLES, dtype=bool), 0, coeff)
        
        # grad_pair[i] = sum_j coeff[i, j] * diff[i, j]
        grad_pair = np.sum(coeff[:, :, np.newaxis] * diff, axis=1)
        
        grad += grad_pair
        
        # Update
        v = momentum * v + (1 - momentum) * grad
        pts = pts + lr * v
        
        # Keep points inside the square?
        # Clipping can cause issues with gradients, but ensures validity.
        # However, if we clip, the gradient might push them back out immediately.
        # Better to rely on the soft constraint.
        # But if they go far out, soft_min becomes very negative.
        # Let's do a hard clip to [-0.5, 0.5] to prevent explosion, 
        # but this might create "sticking" at boundaries.
        # A softer approach: penalize going out strongly.
        # The soft-min already does this.
        
        # Evaluate true radius for best tracking
        # This is expensive O(N^2). Do it less frequently?
        # Or just use the soft_min value as a proxy?
        # The soft_min is a lower bound on the true min? No, it's an approximation.
        # Let's compute the true radius every 100 steps.
        
        if steps % 50 == 0:
            val = _compute_radius(pts)
            if val > best_val:
                best_val = val
                best_pts = pts.copy()
        
        # Adaptive learning rate?
        # If we are stuck, decrease lr.
        # If we are improving fast, maybe increase?
        # Let's decay lr slowly.
        lr = max(0.001, lr * 0.999)
        
        steps += 1
        
        # Check time
        if time.time() > deadline:
            break
            
    # Final evaluation
    final_val = _compute_radius(pts)
    if final_val > best_val:
        best_val = final_val
        best_pts = pts.copy()
        
    return best_pts

def construct():
    """
    Main entry point.
    Returns array of shape (N, 2) with centers of N equal circles.
    """
    # Generate initial configuration
    pts = _init_hex_grid()
    
    # Run optimization
    # Use most of the time budget for optimization
    time_left = _time_remaining()
    # Reserve 5s for safety
    opt_time = max(1.0, time_left - 5.0)
    
    optimized_pts = _optimize(pts, opt_time)
    
    # Final check and return
    # Ensure shape is correct
    if optimized_pts.shape != (N_CIRCLES, 2):
        # Fallback
        m = int(np.ceil(np.sqrt(N_CIRCLES)))
        x = (np.arange(m) + 0.5) / m - 0.5
        y = (np.arange(m) + 0.5) / m - 0.5
        X, Y = np.meshgrid(x, y)
        optimized_pts = np.column_stack([X.ravel(), Y.ravel()])[:N_CIRCLES]
        
    return optimized_pts

# EVOLVE-BLOCK-END
