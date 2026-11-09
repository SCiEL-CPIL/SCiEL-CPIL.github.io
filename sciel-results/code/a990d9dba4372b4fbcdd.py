import numpy as np
import time

N = 100

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin
    
    # 1. Initial Grid with Jitter
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    # Create a grid of m x m x m points
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    centers = grid[:N].copy()
    
    # Add jitter to break symmetry
    np.random.seed(42) # Fixed seed for reproducibility in dev, but evaluator might want variety? 
    # The task says "maximize", usually deterministic is fine unless it's a random search.
    # Let's use a small jitter based on the grid spacing.
    spacing = 1.0 / m
    centers += np.random.uniform(-spacing*0.1, spacing*0.1, size=centers.shape)
    centers = np.clip(centers, -0.5, 0.5)
    
    def calc_score(c):
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists)
        
        wall_dists = 0.5 - np.abs(c)
        min_wall_dist = np.min(wall_dists)
        
        return min(min_pair_dist / 2.0, min_wall_dist)
    
    best_centers = centers.copy()
    best_score = calc_score(best_centers)
    
    # Helper to find min pair and min wall
    def get_min_constraints(c):
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        
        # Find min pair
        min_val = np.min(dists)
        idx = np.unravel_index(np.argmin(dists), dists.shape)
        i, j = idx[0], idx[1]
        
        # Find min wall
        wall_dists = 0.5 - np.abs(c)
        min_wall_val = np.min(wall_dists)
        wall_idx = np.unravel_index(np.argmin(wall_dists), wall_dists.shape)
        wi, wk = wall_idx[0], wall_idx[1]
        
        return min_val, i, j, min_wall_val, wi, wk

    # 2. Push-Apart Phase
    # This phase specifically targets the most violated constraints.
    current_centers = best_centers.copy()
    step_size = 0.5 # Fraction of the violation to correct
    max_iter_push = 5000
    
    for k in range(max_iter_push):
        if time.time() > deadline - 20.0: # Reserve time for other phases
            break
            
        min_pair_dist, i, j, min_wall_dist, wi, wk = get_min_constraints(current_centers)
        
        # Determine which constraint is tighter
        # We want to maximize min(min_pair/2, min_wall)
        # If min_pair/2 < min_wall, push pair apart.
        # Else, push wall point away.
        
        r_pair = min_pair_dist / 2.0
        r_wall = min_wall_dist
        
        if r_pair < r_wall:
            # Push i and j apart
            v = current_centers[i] - current_centers[j]
            norm_v = np.linalg.norm(v)
            if norm_v > 1e-10:
                direction = v / norm_v
                # Move each point by half the step? Or move both?
                # We want to increase distance.
                # Shift = step_size * direction
                # current_centers[i] += shift
                # current_centers[j] -= shift
                shift = step_size * direction
                current_centers[i] += shift
                current_centers[j] -= shift
        else:
            # Push point wi away from wall wk
            # Wall is at 0.5 or -0.5
            val = current_centers[wi, wk]
            if val > 0:
                # Wall is at 0.5, push left (negative direction)
                direction = -1.0
            else:
                # Wall is at -0.5, push right (positive direction)
                direction = 1.0
            
            current_centers[wi, wk] += step_size * direction
            
        # Clamp to cube
        current_centers = np.clip(current_centers, -0.5, 0.5)
        
        # Check score
        score = calc_score(current_centers)
        if score > best_score:
            best_score = score
            best_centers = current_centers.copy()
            
        # Decay step size
        step_size *= 0.999

    # 3. Gradient Ascent on Soft-Min Surrogate
    # Objective: Maximize SoftMin(0.5 * pair_dists, wall_dists)
    # SoftMin(x) = -1/alpha * log(sum(exp(-alpha * x)))
    # Gradient of SoftMin w.r.t x_k is proportional to exp(-alpha * x_k) / sum(exp(-alpha * x_j))
    # This gives weights to the "active" constraints (those close to the minimum).
    
    current_centers = best_centers.copy()
    T = 0.1 # Temperature for soft-min (alpha = 1/T)
    lr = 0.001
    
    # Precompute indices for pairs
    i_indices, j_indices = np.triu_indices(N, k=1)
    pairs = (i_indices, j_indices)
    
    while time.time() < deadline:
        if T < 0.01:
            break
            
        c = current_centers
        
        # Calculate distances and wall distances
        # Pair distances
        diff = c[i_indices] - c[j_indices]
        pair_dists = np.linalg.norm(diff, axis=1)
        pair_r = 0.5 * pair_dists
        
        # Wall distances
        wall_dists = 0.5 - np.abs(c)
        # Flatten wall dists to match shape for sum
        # We have N*3 wall distances
        wall_flat = wall_dists.flatten()
        
        # Combine all constraints into one vector
        # Constraints: pair_r (N(N-1)/2) and wall_flat (3N)
        # We want to maximize the soft-min of these.
        
        # To avoid underflow in exp, we can subtract the max? 
        # Actually, we want exp(-alpha * x). If x is large, exp is small.
        # If x is small (close to 0), exp is large.
        # The soft-min is dominated by the smallest x.
        
        alpha = 1.0 / T
        # Compute weights
        # w_k = exp(-alpha * x_k)
        # We need to normalize these to get the gradient weights.
        
        # Optimization: 
        # log_sum_exp(-alpha * x)
        # max_x = max(x) -> exp(-alpha * max_x) is the smallest term? No.
        # If x is small, -alpha * x is large positive? No, alpha > 0.
        # If x is small (bad), -alpha * x is less negative (larger) than -alpha * x_large.
        # So exp(-alpha * x_small) > exp(-alpha * x_large).
        # The term with the smallest x has the largest weight.
        
        # Let's compute the weights.
        # x_pairs = pair_r
        # x_walls = wall_flat
        
        # To be numerically stable, we can just compute the weighted average of gradients.
        # Grad Objective = sum_k w_k * Grad x_k
        # where w_k = exp(-alpha * x_k) / sum_j exp(-alpha * x_j)
        
        # Calculate unnormalized weights
        w_pairs = np.exp(-alpha * pair_r)
        w_walls = np.exp(-alpha * wall_flat)
        
        total_w = np.sum(w_pairs) + np.sum(w_walls)
        
        if total_w < 1e-10:
            break
            
        w_pairs /= total_w
        w_walls /= total_w
        
        # Compute Gradient
        # Grad w.r.t c for pair (i, j):
        # x_k = 0.5 * ||c_i - c_j||
        # d(x_k)/d(c_i) = 0.5 * (c_i - c_j) / ||c_i - c_j||
        # d(x_k)/d(c_j) = 0.5 * (c_j - c_i) / ||c_i - c_j||
        
        grad = np.zeros_like(c)
        
        # Pair contributions
        # diff is c_i - c_j
        # dist is pair_dists
        # Avoid div by 0
        safe_dists = np.where(pair_dists > 1e-10, pair_dists, 1.0)
        unit_vecs = diff / safe_dists[:, np.newaxis]
        
        # Contribution to c_i: +0.5 * unit_vec * w_k
        # Contribution to c_j: -0.5 * unit_vec * w_k
        
        # Use np.add.at for accumulation
        grad_contrib = 0.5 * unit_vecs * w_pairs[:, np.newaxis]
        
        np.add.at(grad, i_indices, grad_contrib)
        np.add.at(grad, j_indices, -grad_contrib)
        
        # Wall contributions
        # x_k = 0.5 - |c_{i, k}|
        # d(x_k)/d(c_{i, k}) = -sign(c_{i, k})
        # Note: if c is 0, sign is 0, but gradient is undefined. We can ignore or use 0.
        
        wall_grad = -np.sign(c) * w_walls.reshape(N, 3)
        grad += wall_grad
        
        # Update
        current_centers += lr * grad
        current_centers = np.clip(current_centers, -0.5, 0.5)
        
        # Check score
        score = calc_score(current_centers)
        if score > best_score:
            best_score = score
            best_centers = current_centers.copy()
            
        # Decay learning rate and temperature
        lr *= 0.999
        T *= 0.999
        
    return best_centers
