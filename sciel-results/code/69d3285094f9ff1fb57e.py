import numpy as np
import time

N = 100

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    # 1. Initialization: 5x5x5 grid with jitter
    m = 5
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    centers = grid[:N].copy()
    
    # Add small jitter to break symmetry
    spacing = 1.0 / m
    centers += np.random.uniform(-spacing * 0.05, spacing * 0.05, size=centers.shape)
    centers = np.clip(centers, -0.5, 0.5)

    def compute_score(c):
        """
        Compute the maximum radius for the given centers.
        """
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists)
        
        wall_dists = 0.5 - np.abs(c)
        min_wall_dist = np.min(wall_dists)
        
        return min(min_pair_dist / 2.0, min_wall_dist)

    best_centers = centers.copy()
    best_score = compute_score(best_centers)

    # 2. Push-Apart Phase (Constraint Violation Correction)
    # This phase targets the most violated constraints to quickly improve the initial packing.
    current_centers = best_centers.copy()
    step_size = 0.1
    max_iter_push = 2000
    
    for k in range(max_iter_push):
        if time.time() > deadline - 30.0:  # Reserve time for optimization
            break
            
        # Find min pair and min wall
        diff = current_centers[:, np.newaxis, :] - current_centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        min_pair_val = np.min(dists)
        i, j = np.unravel_index(np.argmin(dists), dists.shape)
        
        wall_dists = 0.5 - np.abs(current_centers)
        min_wall_val = np.min(wall_dists)
        wi, wk = np.unravel_index(np.argmin(wall_dists), wall_dists.shape)
        
        r_pair = min_pair_val / 2.0
        r_wall = min_wall_val
        
        if r_pair < r_wall:
            # Push i and j apart
            v = current_centers[i] - current_centers[j]
            norm_v = np.linalg.norm(v)
            if norm_v > 1e-10:
                direction = v / norm_v
                shift = step_size * direction
                current_centers[i] += shift
                current_centers[j] -= shift
        else:
            # Push point wi away from wall wk
            val = current_centers[wi, wk]
            if val > 0:
                direction = -1.0
            else:
                direction = 1.0
            current_centers[wi, wk] += step_size * direction
        
        # Clamp to cube
        current_centers = np.clip(current_centers, -0.5, 0.5)
        
        score = compute_score(current_centers)
        if score > best_score:
            best_score = score
            best_centers = current_centers.copy()
            
        # Decay step size
        step_size *= 0.99

    # 3. Gradient Ascent on Soft-Min Surrogate
    # Objective: Maximize SoftMin(0.5 * pair_dists, wall_dists)
    # SoftMin(x) = -1/alpha * log(sum(exp(-alpha * x)))
    
    current_centers = best_centers.copy()
    
    # Precompute indices for pairs
    i_indices, j_indices = np.triu_indices(N, k=1)
    
    # Adaptive Temperature (alpha)
    # Start with a higher alpha (sharper min) since we already did push-apart,
    # but keep it smooth enough to navigate.
    # The inspiration suggests adaptive beta (alpha).
    alpha_start = 50.0  # Sharper than 10, but not infinite
    alpha_end = 200.0   # Very sharp at the end
    
    lr_start = 0.005
    lr_end = 0.0001
    
    # Number of iterations
    # We will run until deadline
    iter_count = 0
    total_iters = 5000  # Estimate, we'll break on time
    
    while time.time() < deadline - 2.0:
        iter_count += 1
        
        # Interpolate alpha and learning rate
        progress = min(1.0, iter_count / total_iters)
        alpha = alpha_start + (alpha_end - alpha_start) * progress
        lr = lr_start + (lr_end - lr_start) * progress
        
        c = current_centers
        
        # Calculate pair distances and wall distances
        diff = c[i_indices] - c[j_indices]
        pair_dists = np.linalg.norm(diff, axis=1)
        pair_r = 0.5 * pair_dists
        
        wall_dists = 0.5 - np.abs(c)
        wall_flat = wall_dists.flatten()
        
        # Combine constraints
        # We want to maximize the soft-min of all constraints (pair_r and wall_flat)
        
        # Compute weights for soft-min
        # w_k = exp(-alpha * x_k) / sum(exp(-alpha * x_j))
        # To avoid underflow/overflow, we can subtract the min? 
        # No, exp(-alpha * x) is largest when x is smallest.
        # Let's just compute directly. If values are large, exp is small.
        
        # Optimization: Subtract the minimum value to keep exponents manageable?
        # If x_min is the smallest, exp(-alpha * x_min) is the largest.
        # Let's subtract x_min from all x to make the largest exponent 0.
        x_pairs = pair_r
        x_walls = wall_flat
        
        # Combine into one array for weight calculation? 
        # It's easier to compute weights separately and normalize.
        
        # Weights for pairs
        # To prevent underflow if alpha is large and x is large:
        # We can subtract the min of all constraints from x before exp.
        min_constraint = np.min(np.concatenate([x_pairs, x_walls]))
        
        # Shifted values
        x_pairs_shifted = x_pairs - min_constraint
        x_walls_shifted = x_walls - min_constraint
        
        # Now max exponent is 0.
        w_pairs_unnorm = np.exp(-alpha * x_pairs_shifted)
        w_walls_unnorm = np.exp(-alpha * x_walls_shifted)
        
        total_w = np.sum(w_pairs_unnorm) + np.sum(w_walls_unnorm)
        
        if total_w < 1e-10:
            # All weights underflowed, which shouldn't happen with shifting, 
            # but if it does, break or reset.
            break
            
        w_pairs = w_pairs_unnorm / total_w
        w_walls = w_walls_unnorm / total_w
        
        # Compute Gradient
        grad = np.zeros_like(c)
        
        # Pair contributions
        # d/d(c_i) (0.5 * ||c_i - c_j||) = 0.5 * (c_i - c_j) / ||c_i - c_j||
        # d/d(c_j) (0.5 * ||c_i - c_j||) = 0.5 * (c_j - c_i) / ||c_i - c_j|| = -0.5 * (c_i - c_j) / ||c_i - c_j||
        
        safe_dists = np.where(pair_dists > 1e-10, pair_dists, 1.0)
        unit_vecs = diff / safe_dists[:, np.newaxis]
        
        # Contribution to c_i: +0.5 * unit_vec * w_k
        # Contribution to c_j: -0.5 * unit_vec * w_k
        
        grad_contrib = 0.5 * unit_vecs * w_pairs[:, np.newaxis]
        
        np.add.at(grad, i_indices, grad_contrib)
        np.add.at(grad, j_indices, -grad_contrib)
        
        # Wall contributions
        # Wall distance for point i, dim k: 0.5 - |c_ik|
        # d/d(c_ik) (0.5 - |c_ik|) = -sign(c_ik)
        
        # w_walls is flattened (N, 3) -> (N*3,)
        # We need to reshape w_walls back to (N, 3) to apply signs
        w_walls_reshaped = w_walls.reshape(N, 3)
        
        signs = np.sign(c)
        # If c_ik is 0, sign is 0, derivative is undefined but effectively 0 in this context?
        # Actually, at 0, the function is not differentiable. But for optimization, 
        # we can treat sign(0) as 0 or use a small epsilon. 
        # If c_ik is close to 0, the wall constraint is not active (unless radius is huge), 
        # so weight should be low.
        
        # Gradient contribution for wall constraint: -sign(c_ik) * w_k
        grad_wall = -signs * w_walls_reshaped
        
        grad += grad_wall
        
        # Update
        current_centers += lr * grad
        
        # Clamp to cube
        current_centers = np.clip(current_centers, -0.5, 0.5)
        
        # Check score
        score = compute_score(current_centers)
        if score > best_score:
            best_score = score
            best_centers = current_centers.copy()

    # 4. Final Polish: Local Search with Small Steps
    # Use the best centers found so far
    current_centers = best_centers.copy()
    
    # Simple random local search to escape any tiny local minima
    step_size = 0.001
    for _ in range(5000):
        if time.time() > deadline - 0.5:
            break
            
        # Pick a random point
        idx = np.random.randint(N)
        
        # Random direction
        direction = np.random.normal(size=3)
        direction /= np.linalg.norm(direction)
        
        # Try moving
        new_centers = current_centers.copy()
        new_centers[idx] += step_size * direction
        new_centers = np.clip(new_centers, -0.5, 0.5)
        
        score = compute_score(new_centers)
        if score > best_score:
            best_score = score
            best_centers = new_centers.copy()
            current_centers = new_centers.copy()
        else:
            # Try opposite direction?
            new_centers = current_centers.copy()
            new_centers[idx] -= step_size * direction
            new_centers = np.clip(new_centers, -0.5, 0.5)
            score = compute_score(new_centers)
            if score > best_score:
                best_score = score
                best_centers = new_centers.copy()
                current_centers = new_centers.copy()
        
        # Decay step size slowly
        step_size *= 0.9999

    return best_centers
