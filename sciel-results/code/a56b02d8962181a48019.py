import numpy as np
import time

N = 100

def construct():
    start_time = time.time()
    time_limit = 140.0  # Leave 10s margin for safety (150s total)
    
    best_r = 0.0
    best_centers = None
    
    def get_min_radius(centers):
        """
        Calculate the exact maximum radius for a given configuration.
        This is the score function.
        """
        # Compute pairwise distances
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dist, np.inf)
        min_pair_dist = np.min(dist)
        
        # Wall distances
        wall_dists = 0.5 - np.abs(centers)
        min_wall_dist = np.min(wall_dists)
        
        return min(min_pair_dist / 2.0, min_wall_dist)

    def optimize_soft(centers, max_steps=20000, beta_start=10.0, beta_end=1000.0, lr_start=0.01, lr_end=0.0001):
        """
        Optimizes the configuration by maximizing a soft-min surrogate of the radius.
        This provides smooth gradients from all constraints, helping escape local minima
        that are stuck due to non-smoothness of the hard min function.
        """
        centers = centers.copy()
        current_r = get_min_radius(centers)
        best_local_r = current_r
        best_local_centers = centers.copy()
        
        # Precompute indices for vectorization
        # We need to compute gradients of S = -1/beta * ln(sum(exp(-beta * d)))
        # d includes pair distances and wall distances.
        
        # To save time, we will compute the gradient in a loop but vectorize the inner parts.
        # However, for N=100, full pairwise matrices are cheap (100x100 = 10k ops).
        
        for step in range(max_steps):
            if time.time() - start_time > time_limit:
                break
            
            # Annealing parameters
            progress = step / max_steps
            beta = beta_start * (beta_end / beta_start) ** progress
            lr = lr_start * (lr_end / lr_start) ** progress
            
            # 1. Compute all pairwise distances and wall distances
            diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
            dist_mat = np.linalg.norm(diff, axis=2)
            np.fill_diagonal(dist_mat, 1e10) # Avoid self-interaction
            
            # Wall distances: distance from center to each of the 6 walls
            # Walls are at -0.5 and 0.5 for x, y, z
            # dist to -0.5 wall: center + 0.5
            # dist to 0.5 wall: 0.5 - center
            wall_dists = np.empty((N, 6))
            wall_dists[:, 0] = centers[:, 0] + 0.5
            wall_dists[:, 1] = 0.5 - centers[:, 0]
            wall_dists[:, 2] = centers[:, 1] + 0.5
            wall_dists[:, 3] = 0.5 - centers[:, 1]
            wall_dists[:, 4] = centers[:, 2] + 0.5
            wall_dists[:, 5] = 0.5 - centers[:, 2]
            
            # 2. Compute weights for soft-min
            # We want to maximize min(d). 
            # Surrogate: S = -1/beta * ln(sum(w_k * exp(-beta * d_k)))
            # Gradient dS/da = -1/beta * (1/sum) * sum( (-beta * dw_k/da) * exp(-beta * d_k) )
            #                = sum( (dw_k/da) * exp(-beta * d_k) ) / sum( exp(-beta * d_k) )
            
            # Weights for pairs: w_k = 1
            # Weights for walls: w_k = 1
            # Note: The radius is min(min_pair/2, min_wall).
            # To be consistent, we should scale pair distances by 1/2 in the exponent?
            # Or just optimize min_pair and min_wall separately and take min?
            # Better: Optimize min(min_pair, 2*min_wall) effectively.
            # Let's define "effective distances" d_eff.
            # d_pair_eff = dist_mat
            # d_wall_eff = 2 * wall_dists
            # Then r = 0.5 * min(d_eff).
            # Maximizing min(d_eff) is equivalent to maximizing r.
            
            d_pair_eff = dist_mat
            d_wall_eff = 2.0 * wall_dists
            
            # Flatten for calculation
            # Mask for pairs: only i < j to avoid double counting, but gradient accumulates symmetrically
            # Actually, sum over all i!=j is fine, it just doubles the gradient magnitude for pairs,
            # which is equivalent to adjusting beta or lr. Let's stick to i<j for clarity and efficiency.
            
            # Create upper triangular mask
            mask = np.triu(np.ones((N, N), dtype=bool), k=1)
            
            # Extract distances
            pair_dists = d_pair_eff[mask]
            wall_dists_flat = d_wall_eff.flatten()
            
            # Combine all distances into one array for weight calculation
            # This is a bit expensive to concatenate every step, but N=100 is small.
            # Alternatively, compute weights separately.
            
            # Let's compute weights separately to avoid large temporary arrays
            # P = sum(exp(-beta * d_pair))
            # W = sum(exp(-beta * d_wall))
            # Total = P + W
            
            exp_pairs = np.exp(-beta * pair_dists)
            exp_walls = np.exp(-beta * wall_dists_flat)
            
            sum_pairs = np.sum(exp_pairs)
            sum_walls = np.sum(exp_walls)
            total_sum = sum_pairs + sum_walls
            
            if total_sum == 0:
                break
                
            # Weights
            w_pairs = exp_pairs / total_sum
            w_walls = exp_walls / total_sum
            
            # 3. Compute Gradient
            # grad = sum_k w_k * grad_d_k
            # 
            # For a pair (i, j) with distance d_ij:
            # d_ij = ||c_i - c_j||
            # grad_c_i d_ij = (c_i - c_j) / d_ij
            # grad_c_j d_ij = (c_j - c_i) / d_ij = - grad_c_i
            #
            # For a wall distance (e.g., 0.5 - c_ix):
            # d = 0.5 - c_ix
            # grad_c_ix d = -1
            # d_eff = 2*d -> grad_c_ix d_eff = -2
            # For c_ix + 0.5: grad_c_ix d_eff = 2
            
            grad = np.zeros_like(centers)
            
            # Pair contributions
            # We have w_pairs corresponding to the flattened mask.
            # We need to scatter these gradients back to the (N, N) matrix structure or loop.
            # Since N=100, looping over the ~4950 pairs is fast in Python if we vectorize the accumulation?
            # No, let's use matrix operations.
            
            # Create a matrix of weights W_pair (N, N)
            W_pair_mat = np.zeros((N, N))
            W_pair_mat[mask] = w_pairs
            
            # Term: sum_j w_ij * (c_i - c_j) / d_ij
            # Let U_ij = w_ij / d_ij
            # grad_c_i = sum_j U_ij * (c_i - c_j) = c_i * sum_j U_ij - sum_j U_ij * c_j
            # Note: d_ij is symmetric.
            
            U_pair = np.zeros((N, N))
            # Avoid division by zero (though d is inf on diagonal, and we only look at i!=j)
            # d_pair_eff has inf on diagonal.
            # We only care about i!=j where mask is true.
            # Initialize U with 0.
            # For i!=j: U_ij = w_ij / d_ij
            
            # We need to divide w_pairs (flattened) by d_pair_eff (flattened).
            u_flat = w_pairs / pair_dists
            U_pair[mask] = u_flat
            
            # Symmetrize? No, the gradient calculation handles direction.
            # grad_c_i = sum_j U_ij (c_i - c_j)
            # = c_i * sum_j U_ij - sum_j U_ij c_j
            
            row_sums_U = np.sum(U_pair, axis=1)
            weighted_c = U_pair @ centers
            
            grad_pairs = centers * row_sums_U[:, np.newaxis] - weighted_c
            
            # Wall contributions
            # w_walls is flattened (N, 6).
            # Shape: (N, 6)
            W_wall_mat = w_walls.reshape(N, 6)
            
            # Gradients for each wall component:
            # 0: c_x + 0.5 -> d_eff = 2(c_x+0.5) -> grad = 2
            # 1: 0.5 - c_x -> d_eff = 2(0.5-c_x) -> grad = -2
            # 2: c_y + 0.5 -> grad = 2
            # 3: 0.5 - c_y -> grad = -2
            # 4: c_z + 0.5 -> grad = 2
            # 5: 0.5 - c_z -> grad = -2
            
            # grad_c_x = 2 * W_wall[:, 0] - 2 * W_wall[:, 1]
            # grad_c_y = 2 * W_wall[:, 2] - 2 * W_wall[:, 3]
            # grad_c_z = 2 * W_wall[:, 4] - 2 * W_wall[:, 5]
            
            grad_walls = np.zeros_like(centers)
            grad_walls[:, 0] = 2.0 * W_wall_mat[:, 0] - 2.0 * W_wall_mat[:, 1]
            grad_walls[:, 1] = 2.0 * W_wall_mat[:, 2] - 2.0 * W_wall_mat[:, 3]
            grad_walls[:, 2] = 2.0 * W_wall_mat[:, 4] - 2.0 * W_wall_mat[:, 5]
            
            grad = grad_pairs + grad_walls
            
            # 4. Update
            # Normalize gradient to prevent huge steps if one constraint is very tight
            grad_norm = np.linalg.norm(grad)
            if grad_norm > 0:
                grad = grad / grad_norm
            
            centers += lr * grad
            
            # Clamp to box to ensure validity (though gradient should keep them inside)
            centers = np.clip(centers, -0.49999, 0.49999)
            
            # 5. Evaluate
            new_r = get_min_radius(centers)
            
            if new_r > best_local_r:
                best_local_r = new_r
                best_local_centers = centers.copy()
                
            # If we are stuck for a while, add a tiny random kick to escape local minima
            # This is a simple heuristic
            if step % 500 == 0 and step > 0:
                # Check if improvement has been minimal recently? 
                # For simplicity, just add a small noise scaled by lr
                centers += np.random.randn(N, 3) * lr * 0.5
                centers = np.clip(centers, -0.49999, 0.49999)

        return best_local_centers, best_local_r

    def optimize_hard(centers, max_steps=5000, step_size=0.002):
        """
        Final polish using hard constraint logic (force-directed) to fix precision issues
        or fine-tune the tightest constraints.
        """
        centers = centers.copy()
        current_r = get_min_radius(centers)
        best_local_r = current_r
        best_local_centers = centers.copy()
        
        step = step_size
        
        for i in range(max_steps):
            if time.time() - start_time > time_limit:
                break
                
            diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
            dist = np.linalg.norm(diff, axis=2)
            np.fill_diagonal(dist, np.inf)
            
            # Find min pair
            min_idx = np.unravel_index(np.argmin(dist), dist.shape)
            i_min, j_min = min_idx
            min_dist_val = dist[i_min, j_min]
            
            # Find min wall
            wall_dists = 0.5 - np.abs(centers)
            min_wall_val = np.min(wall_dists)
            min_wall_idx = np.unravel_index(np.argmin(wall_dists), wall_dists.shape)
            i_wall, k_wall = min_wall_idx
            
            # Determine which is the bottleneck
            r_pair = min_dist_val / 2.0
            r_wall = min_wall_val
            
            forces = np.zeros_like(centers)
            
            # Apply force to the bottleneck
            if r_pair < r_wall:
                # Push pair apart
                direction = (centers[i_min] - centers[j_min]) / (min_dist_val + 1e-10)
                # Push slightly more than the deficit
                push = 0.5 * step
                forces[i_min] += direction * push
                forces[j_min] -= direction * push
            else:
                # Push away from wall
                # Determine sign
                if centers[i_wall, k_wall] > 0:
                    forces[i_wall, k_wall] -= 0.5 * step
                else:
                    forces[i_wall, k_wall] += 0.5 * step
            
            # Add weak global repulsion to prevent clustering
            # Only if we have time? This is O(N^2).
            # For polish, we assume the structure is good, just need to nudge.
            
            centers += forces
            centers = np.clip(centers, -0.49999, 0.49999)
            
            new_r = get_min_radius(centers)
            
            if new_r > best_local_r:
                best_local_r = new_r
                best_local_centers = centers.copy()
            else:
                # If no improvement, shrink step
                step *= 0.9
                if step < 1e-6:
                    break

        return best_local_centers, best_local_r

    # --- Main Loop ---
    
    # 1. Cubic Grid
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    init_grid = grid[:N]
    
    c1, r1 = optimize_soft(init_grid, max_steps=15000)
    if r1 > best_r:
        best_r = r1
        best_centers = c1
        
    # Polish
    if time.time() - start_time < time_limit - 20:
        c1_pol, r1_pol = optimize_hard(c1, max_steps=2000)
        if r1_pol > best_r:
            best_r = r1_pol
            best_centers = c1_pol

    # 2. FCC-like Initialization
    s = 0.3
    pts = []
    for i in range(3):
        for j in range(3):
            for k in range(3):
                base = np.array([i, j, k]) * s
                pts.append(base)
                pts.append(base + np.array([s/2, s/2, 0]))
                pts.append(base + np.array([s/2, 0, s/2]))
                pts.append(base + np.array([0, s/2, s/2]))
    
    fcc_grid = np.array(pts)
    fcc_grid -= fcc_grid.mean(axis=0)
    extent = np.max(np.abs(fcc_grid))
    if extent > 0.5:
        fcc_grid *= (0.5 / extent)
    
    dists_from_origin = np.linalg.norm(fcc_grid, axis=1)
    order = np.argsort(dists_from_origin)
    init_fcc = fcc_grid[order[:N]]
    
    if time.time() - start_time < time_limit - 40:
        c2, r2 = optimize_soft(init_fcc, max_steps=15000)
        if r2 > best_r:
            best_r = r2
            best_centers = c2
            
        if time.time() - start_time < time_limit - 20:
            c2_pol, r2_pol = optimize_hard(c2, max_steps=2000)
            if r2_pol > best_r:
                best_r = r2_pol
                best_centers = c2_pol

    # 3. Random Initialization
    np.random.seed(42)
    init_random = np.random.uniform(-0.4, 0.4, (N, 3))
    
    if time.time() - start_time < time_limit - 40:
        c3, r3 = optimize_soft(init_random, max_steps=15000)
        if r3 > best_r:
            best_r = r3
            best_centers = c3
            
        if time.time() - start_time < time_limit - 20:
            c3_pol, r3_pol = optimize_hard(c3, max_steps=2000)
            if r3_pol > best_r:
                best_r = r3_pol
                best_centers = c3_pol

    # 4. Final Polish on the best found
    if best_centers is not None and time.time() - start_time < time_limit - 5:
        c_best, r_best = optimize_hard(best_centers, max_steps=3000, step_size=0.001)
        if r_best > best_r:
            best_r = r_best
            best_centers = c_best

    if best_centers is None:
        return init_grid

    return best_centers
