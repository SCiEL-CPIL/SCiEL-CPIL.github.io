import numpy as np
import time
import math
from scipy.spatial import cKDTree
from scipy.optimize import minimize

N = 200

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin

    def compute_radius(centers):
        """Compute the exact maximum radius for given centers."""
        wall_dists = 0.5 - np.abs(centers)
        min_wall = np.min(wall_dists)
        
        if len(centers) < 2:
            return min_wall
            
        tree = cKDTree(centers)
        dists, _ = tree.query(centers, k=2)
        min_pair = np.min(dists[:, 1])
        
        r = min(min_wall, min_pair / 2.0)
        return max(0.0, r)

    def soft_min_objective_and_grad(centers, beta=100.0):
        """
        Compute a smooth approximation of the radius objective and its gradient.
        We maximize r, so we minimize -r.
        r_soft = soft_min( [min_pair/2, min_wall] )
        Actually, we want to maximize the minimum of:
        1. Distances to walls: 0.5 - |x_i|
        2. Pairwise distances / 2: ||x_i - x_j|| / 2
        
        Let f_k be the set of constraints (distances).
        soft_min(f) = -1/beta * log(sum(exp(-beta * f_k)))
        This is a smooth lower bound on min(f_k). As beta -> inf, it converges to min(f_k).
        
        Objective to minimize: -soft_min(f)
        """
        # 1. Wall distances
        # f_wall_i = 0.5 - max(|x_i|, |y_i|)  -- Wait, distance to square boundary is min(0.5-x, 0.5+x, 0.5-y, 0.5+y)
        # For a point (x,y) in [-0.5, 0.5], dist to boundary = min(0.5-x, 0.5+x, 0.5-y, 0.5+y)
        # This is equivalent to 0.5 - max(|x|, |y|).
        
        abs_c = np.abs(centers)
        max_abs = np.max(abs_c, axis=1)
        wall_dists = 0.5 - max_abs
        
        # 2. Pairwise distances
        # We only need to consider pairs that are close to the minimum to keep computation tractable,
        # but for gradient accuracy, we need the "active" set.
        # A full O(N^2) gradient is too slow for 200 points (40k pairs).
        # Strategy: Use a KDTree to find K nearest neighbors for each point. 
        # Assume only these pairs contribute significantly to the soft min.
        # Let K=10.
        
        K = 10
        tree = cKDTree(centers)
        dists, idxs = tree.query(centers, k=K+1)
        # dists[:, 0] is 0. We take dists[:, 1:]
        # idxs[:, 1:]
        
        # To avoid double counting and ensure symmetry, we can just sum over all i and their K neighbors.
        # The gradient will be the sum of gradients from each pair.
        
        # Vectorized computation of pair distances
        # i_indices: [0, 0, ..., 1, 1, ...]
        # j_indices: [neighbors of 0, ..., neighbors of 1, ...]
        
        # Flatten for vectorization
        # This creates a large array of size N*K. 200*10 = 2000 pairs. This is very fast.
        
        i_idx = np.repeat(np.arange(N), K)
        j_idx = idxs[:, 1:].flatten()
        
        # Filter out self (shouldn't be there if k=K+1 and we skip 0, but just in case)
        # Also, we might have duplicates if a point is a neighbor to multiple points, 
        # but that's fine, the gradient just adds up.
        
        # Calculate distances
        diffs = centers[i_idx] - centers[j_idx]
        pair_dists = np.linalg.norm(diffs, axis=1)
        
        # The constraint value is dist / 2
        pair_constraints = pair_dists / 2.0
        
        # Combine wall and pair constraints
        # We want soft_min of all these values.
        # To avoid numerical issues with exp, we subtract the max (or a reference)
        # But soft_min is -1/beta * log(sum(exp(-beta * f)))
        # If f is small (e.g., 0.03), -beta*f is large negative, exp is small.
        # It's safer to compute:
        # m = min(all_f)
        # soft_min = m - 1/beta * log(sum(exp(-beta * (f_k - m))))
        
        all_f = np.concatenate([wall_dists, pair_constraints])
        
        m = np.min(all_f)
        
        # Calculate log-sum-exp term
        # exp(-beta * (f_k - m))
        # Since f_k >= m, (f_k - m) >= 0, so exponent is <= 0. Safe.
        exp_terms = np.exp(-beta * (all_f - m))
        log_sum_exp = np.log(np.sum(exp_terms))
        
        soft_min_val = m - (1.0 / beta) * log_sum_exp
        
        # Gradient
        # d(soft_min)/df_k = - exp(-beta*(f_k - m)) / sum(exp(-beta*(f_k - m)))
        # Let weights w_k = exp(-beta*(f_k - m)) / sum(exp(...))
        # Then d(soft_min)/d(centers) = sum_k w_k * d(f_k)/d(centers)
        
        weights = exp_terms / np.sum(exp_terms)
        
        # Gradient of wall distances
        # f_wall_i = 0.5 - max(|x_i|, |y_i|)
        # d(f_wall_i)/dx_i = -sign(x_i) if |x_i| > |y_i| else 0
        # d(f_wall_i)/dy_i = -sign(y_i) if |y_i| > |x_i| else 0
        # If |x_i| == |y_i|, it's non-differentiable, but measure zero. We can average or pick one.
        
        grad_centers = np.zeros_like(centers)
        
        # Wall gradients
        # weights[:N] correspond to wall_dists
        w_wall = weights[:N]
        
        # Determine which coordinate is the max
        is_x_max = abs_c[:, 0] > abs_c[:, 1]
        
        # d/dx_i
        dx_wall = np.zeros(N)
        dy_wall = np.zeros(N)
        
        dx_wall[is_x_max] = -np.sign(centers[is_x_max, 0])
        dy_wall[~is_x_max] = -np.sign(centers[~is_x_max, 1])
        
        grad_centers[:, 0] += w_wall * dx_wall
        grad_centers[:, 1] += w_wall * dy_wall
        
        # Pair gradients
        # f_pair_ij = ||c_i - c_j|| / 2
        # d(f_pair_ij)/dc_i = (c_i - c_j) / (2 * ||c_i - c_j||)
        # d(f_pair_ij)/dc_j = (c_j - c_i) / (2 * ||c_i - c_j||)
        
        w_pair = weights[N:]
        
        # diffs = c_i - c_j
        # norm = pair_dists
        # grad_i = diffs / (2 * norm)
        # grad_j = -diffs / (2 * norm)
        
        safe_dists = np.where(pair_dists < 1e-12, 1e-12, pair_dists)
        grad_factor = w_pair / (2.0 * safe_dists)
        
        grad_i = diffs * grad_factor[:, np.newaxis]
        grad_j = -diffs * grad_factor[:, np.newaxis]
        
        # Accumulate gradients
        np.add.at(grad_centers, i_idx, grad_i)
        np.add.at(grad_centers, j_idx, grad_j)
        
        # Objective is -soft_min_val
        # Gradient of objective is -grad(soft_min)
        # So we return -soft_min_val and -grad_centers
        return -soft_min_val, -grad_centers

    def hexagonal_initialization(n_rows, n_cols):
        """Create a hexagonal packing layout."""
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

    def gradient_refinement(centers, max_time, beta_start=50.0, beta_end=500.0):
        """
        Perform gradient ascent on the soft-min objective.
        """
        centers = centers.copy().flatten()
        
        # We'll do a sequence of minimizations with increasing beta
        # and decreasing step size (via L-BFGS-B)
        
        best_centers = centers.copy()
        best_val = np.inf
        
        # Define bounds
        bounds = [(-0.5, 0.5)] * (2 * N)
        
        # We use L-BFGS-B which handles bounds well.
        
        # Schedule of betas
        betas = [50, 100, 200, 400, 800]
        
        for beta in betas:
            if time.time() > max_time:
                break
            
            def fun(x):
                c = x.reshape(N, 2)
                val, grad = soft_min_objective_and_grad(c, beta)
                return val, grad.flatten()
                
            try:
                res = minimize(fun, centers, method='L-BFGS-B', jac=True, bounds=bounds, 
                               options={'maxiter': 1000, 'gtol': 1e-8, 'maxls': 50})
                
                if res.success or res.fun < best_val:
                    centers = res.x
                    best_val = res.fun
                    # Check radius
                    r = compute_radius(centers.reshape(N, 2))
                    # We track the "soft" value for optimization, but radius for comparison
                    # Actually, let's just track the centers that give the best radius
                    # But the optimization minimizes -soft_r. So lower res.fun is better.
                    # However, soft_r is an approximation. 
                    # Let's evaluate the true radius.
                    true_r = compute_radius(centers.reshape(N, 2))
                    
                    # If this true radius is better, update best_centers
                    # We need to compare true_r.
                    # Let's maintain a variable for best true radius.
                    
            except Exception as e:
                pass
                
        # Final evaluation
        final_centers = centers.reshape(N, 2)
        return final_centers

    def cluster_refinement(centers, max_time):
        """
        Identify contact clusters (pairs with distance close to min_dist) and 
        perform local search on the centers.
        This is a structural change: instead of moving one point at a time randomly,
        we move groups of points that are "stuck" together.
        """
        centers = centers.copy()
        current_r = compute_radius(centers)
        best_r = current_r
        best_centers = centers.copy()
        
        # We will iterate:
        # 1. Find the bottleneck distance d_min.
        # 2. Identify all pairs with dist < d_min * 1.05 (active set).
        # 3. For each active pair, try to push them apart slightly and see if it helps.
        #    This is effectively a force-directed relaxation but deterministic and targeted.
        
        for _ in range(50): # Max iterations
            if time.time() > max_time:
                break
                
            tree = cKDTree(centers)
            dists, idxs = tree.query(centers, k=2)
            min_dist = np.min(dists[:, 1])
            r = min_dist / 2.0
            
            # Also check walls
            wall_dists = 0.5 - np.abs(centers)
            min_wall = np.min(wall_dists)
            r = min(r, min_wall)
            
            if r > best_r:
                best_r = r
                best_centers = centers.copy()
                
            # Identify active pairs
            # Pairs where dist is close to min_dist
            active_mask = dists[:, 1] < min_dist * 1.01
            active_i = np.where(active_mask)[0]
            active_j = idxs[active_i, 1]
            
            if len(active_i) == 0:
                break
                
            # Calculate forces
            # For each active pair, apply a repulsive force
            # Also apply attractive force towards center if wall is bottleneck?
            # No, just repulsive between close pairs.
            
            forces = np.zeros_like(centers)
            
            for i, j in zip(active_i, active_j):
                diff = centers[i] - centers[j]
                dist = np.linalg.norm(diff)
                if dist < 1e-9:
                    # Random nudge
                    diff = np.random.randn(2) * 1e-9
                    dist = 1e-9
                force = diff / dist
                # Strength proportional to how close they are
                # We want to increase the distance.
                # Force on i is in direction of diff.
                # Force on j is in direction of -diff.
                strength = 1.0 
                forces[i] += force * strength
                forces[j] -= force * strength
                
            # Wall forces
            # If a point is close to a wall, push it inwards
            for i in range(N):
                x, y = centers[i]
                if 0.5 - abs(x) < min_wall * 1.01:
                    # Close to left or right wall
                    if x > 0: # Right wall
                        forces[i, 0] -= 1.0
                    else: # Left wall
                        forces[i, 0] += 1.0
                if 0.5 - abs(y) < min_wall * 1.01:
                    # Close to top or bottom wall
                    if y > 0: # Top wall
                        forces[i, 1] -= 1.0
                    else: # Bottom wall
                        forces[i, 1] += 1.0
                        
            # Normalize forces
            norm_forces = np.linalg.norm(forces, axis=1, keepdims=True)
            norm_forces[norm_forces == 0] = 1.0
            forces = forces / norm_forces
            
            # Move
            step = 0.01 * min_dist
            new_centers = centers + forces * step
            
            # Clamp to bounds
            new_centers = np.clip(new_centers, -0.5, 0.5)
            
            # Check if improvement
            new_r = compute_radius(new_centers)
            
            if new_r > current_r + 1e-12:
                centers = new_centers
                current_r = new_r
            else:
                # Try smaller step
                step *= 0.5
                new_centers = centers + forces * step
                new_centers = np.clip(new_centers, -0.5, 0.5)
                new_r = compute_radius(new_centers)
                if new_r > current_r + 1e-12:
                    centers = new_centers
                    current_r = new_r
                else:
                    # No improvement, break to avoid oscillation
                    break
                    
        return best_centers, best_r

    # Generate diverse initial configurations
    best_centers = None
    best_radius = 0.0
    
    configs = []
    # Hexagonal layouts
    for n_rows in range(13, 17):
        n_cols = math.ceil(N / n_rows)
        pts = hexagonal_initialization(n_rows, n_cols)
        if pts is not None:
            configs.append(pts)
            
    # Square grid
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    configs.append(grid[:N])
    
    # Optimize each configuration
    for idx, init_pts in enumerate(configs):
        if time.time() > deadline - 30:
            break
            
        # 1. Gradient Refinement (Smooth phase)
        if time.time() < deadline - 10:
            try:
                grad_centers = gradient_refinement(init_pts, deadline - 5)
                r = compute_radius(grad_centers)
                if r > best_radius:
                    best_radius = r
                    best_centers = grad_centers.copy()
            except Exception:
                pass
                
        # 2. Cluster Refinement (Discrete/Force phase)
        if time.time() < deadline - 5:
            try:
                # Use the best of init or grad as starting point for cluster
                start_for_cluster = best_centers if best_centers is not None else init_pts
                polished, r_polished = cluster_refinement(start_for_cluster, deadline - 2)
                if r_polished > best_radius:
                    best_radius = r_polished
                    best_centers = polished.copy()
            except Exception:
                pass

    # Final high-precision polish on the absolute best
    if best_centers is not None and time.time() < deadline - 2:
        try:
            polished, r_polished = cluster_refinement(best_centers, deadline - 1)
            if r_polished > best_radius:
                best_radius = r_polished
                best_centers = polished.copy()
        except Exception:
            pass

    if best_centers is None:
        # Fallback
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_centers = grid[:N]
        
    return best_centers
