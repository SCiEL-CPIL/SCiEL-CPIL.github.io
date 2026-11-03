import numpy as np
import time
import math
from scipy.spatial import cKDTree

N = 200

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    def compute_radius(centers):
        """Compute the exact radius efficiently."""
        if not np.all(np.isfinite(centers)):
            return 0.0
        # Check bounds
        if np.any(centers < -0.5) or np.any(centers > 0.5):
            # If out of bounds, the radius is effectively limited by how far out it is? 
            # The contract says: "A centre on or outside the square ... make the candidate invalid"
            # But the score formula is: min(min_pair/2, min_wall_dist)
            # If a center is outside, min_wall_dist is negative.
            # The evaluator likely checks validity first or returns 0/invalid.
            # We should ensure our output is valid.
            pass
            
        wall_dists = 0.5 - np.abs(centers)
        min_wall = np.min(wall_dists)
        
        if min_wall < 0:
            # This configuration is invalid according to the contract ("A centre on or outside...")
            # However, the score formula might just return a negative number or 0.
            # To be safe, we return the computed value which will be negative, 
            # but the evaluator might reject it. 
            # Let's assume the evaluator computes the score as defined:
            # r = min(min_pair/2, min_wall)
            # If min_wall < 0, r < 0. 
            # But the contract says "invalid result... scores 0".
            # So we should avoid returning invalid centers.
            return 0.0

        tree = cKDTree(centers)
        dists, _ = tree.query(centers, k=2)
        min_pair = np.min(dists[:, 1])
        
        r = min(min_wall, min_pair / 2.0)
        return r

    def soft_energy_and_grad(centers, r_current, k=10, softness=0.005):
        """
        Compute a smooth approximation of the objective function and its gradient.
        Objective: Maximize min(distances).
        Equivalent to: Minimize -min(distances).
        We use a soft-min approximation: -softmin(dists).
        softmin(x) ~ -log(sum(exp(-x/T))) * T
        Gradient of softmin(x) w.r.t x_i is -exp(-x_i/T) / sum(exp(-x_j/T))
        
        Here, we want to maximize the minimum of a set of distances (pair and wall).
        Let D be the vector of all relevant distances (pair dists / 2, wall dists).
        We want to maximize min(D).
        This is equivalent to minimizing f(D) = -softmin(D).
        grad f = - grad softmin.
        
        grad softmin(D) w.r.t D_i = -exp(-D_i/T) / sum(exp(-D_j/T))
        So grad f w.r.t D_i = exp(-D_i/T) / sum(exp(-D_j/T))
        
        Then we backpropagate to positions.
        """
        n = len(centers)
        grad = np.zeros_like(centers)
        
        # 1. Wall distances
        # d_wall_i = 0.5 - |c_i|
        # grad d_wall_i w.r.t c_i = -sign(c_i)
        
        # 2. Pair distances
        # d_pair_ij = |c_i - c_j|
        # We consider d_pair_ij / 2 in the objective? 
        # The radius r is min(d_pair/2, d_wall).
        # So the "distances" in the min are d_pair/2 and d_wall.
        
        tree = cKDTree(centers)
        # We need neighbors. Let's use k nearest neighbors to approximate the sum.
        # Actually, for a good gradient, we should consider all pairs? 
        # With N=200, N^2 = 40000 pairs. That's manageable per iteration if vectorized.
        # Let's try full pairwise for accuracy in the gradient, or k-nearest for speed.
        # Given 150s budget, we can afford ~1000-5000 iterations.
        # 40000 ops * 5000 iters = 200M ops. Might be tight in Python loop, but vectorized numpy is fast.
        
        # Let's use vectorized pairwise distance calculation for gradient.
        # diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        # dists = np.linalg.norm(diff, axis=-1)
        
        # To save memory/time, let's stick to k-nearest neighbors for the "soft" part,
        # but this introduces bias. 
        # Alternative: Use the "Repulsion Field" idea but derived from a potential.
        
        # Let's try a simpler potential approach:
        # Energy E = sum_{i<j} 1/d_ij^alpha + sum_i 1/wall_dist_i^alpha
        # This is what the previous program did.
        
        # New Strategy: Langevin Dynamics on a Soft Potential
        # Potential V = sum_{i<j} V_pair(d_ij) + sum_i V_wall(d_wall_i)
        # V_pair(d) = 1/d^2 if d < 2*r_est else 0 (cutoff)
        # V_wall(d) = 1/d^2 if d < r_est else 0
        
        # Force = -grad V
        # Move: c_new = c + force * dt + noise
        
        # This is effectively what the previous program did, but with explicit noise (Langevin).
        # The previous program used "Simulated Annealing Lite" which is similar.
        
        # What is materially different?
        # 1. Different Initialization: Use a "Melting" approach. Start with a dense random cloud,
        #    let repulsion expand it, then cool down.
        # 2. Different Optimization: Use Nesterov Accelerated Gradient Descent (NAG) on the soft objective.
        #    NAG often converges faster than standard GD or SA for smooth approximations.
        
        return None # Placeholder, we will implement NAG in the main loop

    def soft_min_objective_and_grad(centers, T=0.01, k_neighbors=15):
        """
        Compute gradient of -softmin(min_distance).
        We approximate the global min using local k-nearest neighbors to keep it O(N log N).
        """
        n = len(centers)
        grad = np.zeros_like(centers)
        
        # Wall terms
        # d_w_i = 0.5 - |c_i|
        # Contribution to -softmin:
        # Let D be the vector of all candidate radii.
        # We want grad of -softmin(D).
        # softmin(D) = -T * log(sum(exp(-D_i/T)))
        # -softmin(D) = T * log(sum(exp(-D_i/T)))
        # d/dD_i (-softmin) = T * (1/sum) * exp(-D_i/T) * (-1/T) = -exp(-D_i/T) / sum(exp(-D_j/T))
        # Wait, we want to MAXIMIZE min(D). So we MINIMIZE -min(D).
        # Let f(D) = -softmin(D). We want to minimize f.
        # grad f w.r.t D_i = -grad softmin w.r.t D_i
        # grad softmin w.r.t D_i = -exp(-D_i/T) / sum(exp(-D_j/T))
        # So grad f w.r.t D_i = exp(-D_i/T) / sum(exp(-D_j/T))
        # This is a positive weight w_i.
        # Then grad f w.r.t c = sum_i w_i * grad D_i w.r.t c
        
        # We need the sum of exp(-D_j/T) over all j (neighbors + walls).
        # Let's collect all D values and their gradients.
        
        D_vals = []
        grads_D = [] # List of (weight, grad_c)
        
        # Walls
        for i in range(n):
            for dim in range(2):
                c = centers[i, dim]
                if c >= 0:
                    d_w = 0.5 - c
                    grad_c = np.zeros(2)
                    grad_c[dim] = -1.0
                else:
                    d_w = 0.5 + c
                    grad_c = np.zeros(2)
                    grad_c[dim] = 1.0
                
                D_vals.append(d_w)
                grads_D.append(grad_c)
        
        # Pairs (k-nearest)
        tree = cKDTree(centers)
        dists, idxs = tree.query(centers, k=k_neighbors + 1)
        
        for i in range(n):
            for k in range(1, k_neighbors + 1):
                j = idxs[i, k]
                if i == j: continue
                d_vec = centers[i] - centers[j]
                d = dists[i, k]
                # D_val for this pair is d/2
                D_val = d / 2.0
                
                # grad D_val w.r.t c_i = d_vec / d
                # grad D_val w.r.t c_j = -d_vec / d
                if d > 1e-9:
                    grad_i = d_vec / d
                    grad_j = -d_vec / d
                else:
                    grad_i = np.zeros(2)
                    grad_j = np.zeros(2)
                
                D_vals.append(D_val)
                # We need to add gradient contributions to i and j
                # We can't just append a single grad_c here because it affects two centers.
                # We will handle this by storing (center_idx, grad_vec) pairs.
                
                # Let's restructure:
                # We accumulate grad_c[i] += w * grad_i
                # We accumulate grad_c[j] += w * grad_j
                # But we need w (the weight from softmin) first, which requires the sum.
                
                # So we first collect all D_vals, compute weights, then apply.
        
        # To avoid double counting pairs in the sum, we should only count each pair once?
        # Softmin is over the set of all constraints. If we include (i,j) and (j,i), we double the weight.
        # It's better to include each pair once.
        
        # Let's restart the logic for efficiency and correctness.
        
        # 1. Collect unique constraints (d, grad_i, grad_j, i, j)
        #    - Walls: (d_w, grad_c, i, -1)  # -1 means no second center
        #    - Pairs: (d/2, grad_i, grad_j, i, j)
        
        constraints = []
        
        # Walls
        for i in range(n):
            for dim in range(2):
                c = centers[i, dim]
                if c >= 0:
                    d_w = 0.5 - c
                    g = np.zeros(2); g[dim] = -1.0
                else:
                    d_w = 0.5 + c
                    g = np.zeros(2); g[dim] = 1.0
                constraints.append((d_w, i, g, -1, np.zeros(2)))
        
        # Pairs
        for i in range(n):
            for k in range(1, k_neighbors + 1):
                j = idxs[i, k]
                if i < j: # Only count once
                    d_vec = centers[i] - centers[j]
                    d = dists[i, k]
                    D_val = d / 2.0
                    if d > 1e-9:
                        g_i = d_vec / d
                        g_j = -d_vec / d
                    else:
                        g_i = np.zeros(2)
                        g_j = np.zeros(2)
                    constraints.append((D_val, i, g_i, j, g_j))
        
        # 2. Compute weights
        D_arr = np.array([c[0] for c in constraints])
        # Avoid division by zero or overflow
        D_arr = np.maximum(D_arr, 1e-9)
        
        # w_i = exp(-D_i/T) / sum(exp(-D_j/T))
        # To avoid underflow/overflow, shift by min(D)
        D_min = np.min(D_arr)
        exp_terms = np.exp(-(D_arr - D_min) / T)
        sum_exp = np.sum(exp_terms)
        weights = exp_terms / sum_exp
        
        # 3. Accumulate gradient
        total_grad = np.zeros_like(centers)
        for idx, c in enumerate(constraints):
            w = weights[idx]
            i = c[1]
            g_i = c[2]
            j = c[3]
            g_j = c[4]
            
            total_grad[i] += w * g_i
            if j != -1:
                total_grad[j] += w * g_j
        
        # We want to MINIMIZE -softmin, so we move AGAINST the gradient of -softmin?
        # No, we want to MAXIMIZE min(D).
        # Let F(c) = -softmin(D(c)). We want to minimize F.
        # grad F = total_grad calculated above (which is grad of -softmin).
        # So we update c <- c - lr * grad F.
        
        return total_grad

    def nesterov_local_search(centers, max_iters=2000, lr=0.001, T=0.01, k_neighbors=15):
        """
        Nesterov Accelerated Gradient Descent on the soft-min objective.
        """
        centers = centers.copy()
        v = np.zeros_like(centers)
        best_r = compute_radius(centers)
        best_centers = centers.copy()
        
        for it in range(max_iters):
            if time.time() > deadline - 10:
                break
                
            # Compute gradient of -softmin
            grad = soft_min_objective_and_grad(centers, T=T, k_neighbors=k_neighbors)
            
            # Nesterov update
            # y = c + v
            # grad_y = grad_obj(y)  # Ideally, but we approximate using c for speed
            # v_new = 0.9 * v - lr * grad_y
            # c_new = c + v_new
            
            # Simplified Nesterov (Momentum)
            v = 0.9 * v - lr * grad
            centers_new = centers + v
            
            # Clip to bounds to maintain validity
            centers_new = np.clip(centers_new, -0.4999, 0.4999)
            
            r_new = compute_radius(centers_new)
            
            if r_new > best_r:
                best_r = r_new
                best_centers = centers_new.copy()
                
            # Adaptive learning rate / Temperature
            # If we are stuck, increase T or lr?
            # For now, keep fixed.
            
            centers = centers_new
            
            # Anneal T slowly to sharpen the minimum
            if it % 100 == 0:
                T *= 0.95
                lr *= 0.95
                
        return best_centers, best_r

    def hexagonal_initialization(n_rows, n_cols):
        """Create hexagonal packing layout."""
        d = 1.0 / (n_cols + 0.5)
        row_spacing = d * math.sqrt(3) / 2
        
        total_height = (n_rows - 1) * row_spacing + d
        if total_height > 1.0 + 1e-6:
            return None
        
        pts = []
        for row in range(n_rows):
            y = -0.5 + d/2 + row * row_spacing
            if row % 2 == 0:
                n_in_row = n_cols
                x_start = -0.5 + d/2
            else:
                n_in_row = n_cols - 1
                x_start = -0.5 + d
            
            for col in range(n_in_row):
                x = x_start + col * d
                pts.append([x, y])
        
        pts = np.array(pts)
        if len(pts) >= N:
            pts = pts[:N]
        elif len(pts) < N:
            return None
        
        return pts

    def square_grid_init():
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:N]

    # Generate diverse initial configurations
    best_centers = None
    best_radius = 0.0

    # Try multiple hexagonal layouts
    for n_rows in range(13, 17):
        n_cols = math.ceil(N / n_rows)
        pts = hexagonal_initialization(n_rows, n_cols)
        if pts is not None:
            r = compute_radius(pts)
            if r > best_radius:
                best_radius = r
                best_centers = pts.copy()

    # Also try square grid
    grid = square_grid_init()
    r_grid = compute_radius(grid)
    if r_grid > best_radius:
        best_radius = r_grid
        best_centers = grid.copy()

    # Run Nesterov local search
    if best_centers is not None and time.time() < deadline - 20:
        optimized, r_opt = nesterov_local_search(best_centers, max_iters=1000, lr=0.002, T=0.01, k_neighbors=12)
        if r_opt > best_radius:
            best_radius = r_opt
            best_centers = optimized.copy()

    # Second pass with refinement
    if best_centers is not None and time.time() < deadline - 10:
        refined, r_refined = nesterov_local_search(best_centers, max_iters=500, lr=0.0005, T=0.005, k_neighbors=8)
        if r_refined > best_radius:
            best_radius = r_refined
            best_centers = refined.copy()

    if best_centers is None:
        best_centers = square_grid_init()

    return best_centers
