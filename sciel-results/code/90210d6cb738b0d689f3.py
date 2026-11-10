import numpy as np
import time
from scipy.optimize import minimize

N = 100

def construct():
    start_time = time.time()
    time_limit = 145.0  # Leave 5s margin for safety (150s total)
    
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

    def compute_gradients(centers, beta):
        """
        Compute the gradient of the soft-min objective with respect to centers.
        The soft-min objective is: -soft_min = log(sum(exp(-beta * d_i))) / beta - max_d
        We want to maximize soft_min, so we minimize -soft_min.
        The gradient of -soft_min w.r.t. centers is what we return.
        """
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dist, 1e10)
        
        # Wall distances (effective: 2 * wall_dist)
        wall_dists = np.empty((N, 6))
        wall_dists[:, 0] = centers[:, 0] + 0.5
        wall_dists[:, 1] = 0.5 - centers[:, 0]
        wall_dists[:, 2] = centers[:, 1] + 0.5
        wall_dists[:, 3] = 0.5 - centers[:, 1]
        wall_dists[:, 4] = centers[:, 2] + 0.5
        wall_dists[:, 5] = 0.5 - centers[:, 2]
        
        mask = np.triu(np.ones((N, N), dtype=bool), k=1)
        pair_dists = dist[mask]
        wall_flat = 2.0 * wall_dists.flatten()
        
        all_d = np.concatenate([pair_dists, wall_flat])
        max_d = np.max(all_d)
        
        # Numerically stable log-sum-exp
        exp_pairs = np.exp(-beta * (pair_dists - max_d))
        exp_walls = np.exp(-beta * (wall_flat - max_d))
        
        sum_pairs = np.sum(exp_pairs)
        sum_walls = np.sum(exp_walls)
        total = sum_pairs + sum_walls
        
        if total < 1e-300:
            return np.zeros_like(centers)
        
        w_pairs = exp_pairs / total
        w_walls = exp_walls / total
        
        grad = np.zeros_like(centers)
        
        # Pair contributions: d_ij = |c_i - c_j|, d(d_ij)/dc_i = (c_i - c_j)/d_ij
        # The gradient of soft_min w.r.t. c_i involves sum over j of w_ij * (c_i - c_j)/d_ij
        # We're minimizing -soft_min, so the gradient is the negative of that.
        # Actually, let's be careful: soft_min = -log(sum(exp(-beta*d_i)))/beta + max_d
        # d(soft_min)/dc = -1/beta * d/dc log(sum(exp(-beta*d_i)))
        #                = -1/beta * sum_i (-beta * exp(-beta*d_i)/sum) * d(d_i)/dc
        #                = sum_i (exp(-beta*d_i)/sum) * d(d_i)/dc
        # So gradient of soft_min w.r.t. c is sum_i w_i * d(d_i)/dc
        # We want to maximize soft_min, so we move in the direction of this gradient.
        # The scipy minimize function minimizes, so we return -gradient of soft_min.
        
        W_pair_mat = np.zeros((N, N))
        W_pair_mat[mask] = w_pairs
        
        U_pair = np.zeros((N, N))
        U_pair[mask] = w_pairs / (pair_dists + 1e-10)
        
        row_sums_U = np.sum(U_pair, axis=1)
        weighted_c = U_pair @ centers
        
        # grad_pairs[i] = sum_j U_pair[i,j] * (centers[i] - centers[j])
        #               = centers[i] * row_sums_U[i] - weighted_c[i]
        grad_pairs = centers * row_sums_U[:, np.newaxis] - weighted_c
        
        # Wall contributions
        # d(d_wall)/dc for wall k:
        # wall 0 (x+0.5): d/dc_x = 1, so contribution to grad is +w
        # wall 1 (0.5-x): d/dc_x = -1, so contribution to grad is -w
        W_wall_mat = w_walls.reshape(N, 6)
        
        grad_walls = np.zeros_like(centers)
        grad_walls[:, 0] = W_wall_mat[:, 0] - W_wall_mat[:, 1]
        grad_walls[:, 1] = W_wall_mat[:, 2] - W_wall_mat[:, 3]
        grad_walls[:, 2] = W_wall_mat[:, 4] - W_wall_mat[:, 5]
        
        # Total gradient of soft_min w.r.t. centers
        grad_soft_min = grad_pairs + grad_walls
        
        # We want to minimize -soft_min, so return -grad_soft_min
        return -grad_soft_min

    def soft_min_value(centers, beta):
        """Compute the soft-min value (for monitoring)."""
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dist, 1e10)
        
        wall_dists = 0.5 - np.abs(centers)
        d_pair = dist[np.triu_indices(N, k=1)]
        d_wall = 2.0 * wall_dists.flatten()
        
        all_d = np.concatenate([d_pair, d_wall])
        max_d = np.max(all_d)
        log_sum = np.log(np.sum(np.exp(-beta * (all_d - max_d))))
        soft_min = -log_sum / beta + max_d
        
        return soft_min / 2.0  # Convert back to radius scale

    def optimize_active_set(centers, max_iterations=200, tolerance=1e-12):
        """
        Active-set method for the packing problem.
        
        The problem is: maximize r subject to:
          |c_i - c_j| >= 2r for all i < j
          |c_ik| <= 0.5 - r for all i, k
        
        This is equivalent to: maximize r subject to:
          |c_i - c_j|^2 >= (2r)^2 for all i < j
          (0.5 - r - c_ik)^2 >= 0 and c_ik <= 0.5 - r
          (0.5 - r + c_ik)^2 >= 0 and c_ik >= -0.5 + r
        
        Active set method:
        1. Start with current configuration and r = current min radius
        2. Identify active constraints (those at or near the minimum)
        3. Solve a linear system to find a direction that increases r while maintaining active constraints
        4. Take a step in that direction
        5. Repeat until convergence
        """
        centers = centers.copy()
        current_r = get_min_radius(centers)
        best_local_r = current_r
        best_local_centers = centers.copy()
        
        # Define the constraint functions
        # For pair (i,j): g_ij = |c_i - c_j|^2 - (2r)^2 >= 0
        # For wall (i,k,side): g = (0.5 - r - |c_ik|)^2 ... actually let's use:
        # g_ik+ = (0.5 - r) - c_ik >= 0
        # g_ik- = (0.5 - r) + c_ik >= 0
        
        # We'll use a simplified active set approach:
        # 1. Find the most violated constraints
        # 2. Compute a search direction using gradient projection
        
        for iteration in range(max_iterations):
            if time.time() - start_time > time_limit:
                break
            
            # Compute current distances
            diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
            dist = np.linalg.norm(diff, axis=2)
            np.fill_diagonal(dist, np.inf)
            
            wall_dists = 0.5 - np.abs(centers)
            
            # Current radius
            r_pair = np.min(dist) / 2.0
            r_wall = np.min(wall_dists)
            r = min(r_pair, r_wall)
            
            # Find active constraints (within tolerance of minimum)
            tol = max(1e-8, 0.01 * r)
            
            # Active pairs
            pair_mask = dist <= (2 * r + tol)
            np.fill_diagonal(pair_mask, False)
            active_pairs = np.argwhere(pair_mask & np.triu(np.ones((N, N), dtype=bool), k=1))
            
            # Active walls
            active_walls = []
            for i in range(N):
                for k in range(3):
                    if wall_dists[i, k] <= r + tol:
                        # Determine which wall: + or -
                        if centers[i, k] > 0:
                            active_walls.append((i, k, 1))  # positive wall
                        else:
                            active_walls.append((i, k, -1))  # negative wall
            
            # If no active constraints, we can increase r freely (but this shouldn't happen)
            if len(active_pairs) == 0 and len(active_walls) == 0:
                # Just increase r by moving spheres slightly
                break
            
            # Build the constraint Jacobian and solve for a feasible direction
            # We want to find delta_c and delta_r such that:
            #   For each active pair (i,j): (c_i - c_j)·(delta_c_i - delta_c_j) >= 2*r*delta_r
            #   For each active wall (i,k,side): side * delta_c_ik >= -delta_r
            #   delta_r > 0
            #   Minimize ||delta_c||^2 (smallest movement)
            
            # This is a QP: minimize ||delta_c||^2 subject to A*delta >= b*delta_r
            # Let's use a simpler approach: compute the gradient of r w.r.t. centers
            # and project it onto the feasible set
            
            # Compute gradient of r w.r.t. centers
            # r = min(min_pair/2, min_wall)
            # If r = min_pair/2: dr/dc_i = sum over active pairs (i,j) of 0.5 * (c_i - c_j)/|c_i - c_j|
            # If r = min_wall: dr/dc_ik = sign(c_ik) for the active wall
            
            grad_r = np.zeros_like(centers)
            
            if r_pair <= r_wall:
                # r is determined by pairs
                for i, j in active_pairs:
                    d = centers[i] - centers[j]
                    dist_val = np.linalg.norm(d)
                    if dist_val > 1e-10:
                        grad_r[i] += 0.5 * d / dist_val
                        grad_r[j] -= 0.5 * d / dist_val
            else:
                # r is determined by walls
                for i, k, side in active_walls:
                    grad_r[i, k] = side
            
            # Normalize gradient
            grad_norm = np.linalg.norm(grad_r)
            if grad_norm < 1e-12:
                break
            
            grad_r /= grad_norm
            
            # Project gradient onto feasible directions
            # Feasible directions satisfy: for each active constraint, the directional derivative >= 0
            # For pair (i,j): (c_i - c_j)·(d_i - d_j) >= 0  (where d is the direction)
            # For wall (i,k,side): side * d_ik >= 0
            
            # Simple projection: try the gradient direction, and if it violates constraints,
            # reduce the step size
            
            # Try step sizes
            best_step = 0.0
            best_new_r = r
            for scale in [0.1, 0.05, 0.01, 0.005, 0.001]:
                trial_centers = centers + scale * grad_r
                trial_centers = np.clip(trial_centers, -0.5 + 1e-9, 0.5 - 1e-9)
                trial_r = get_min_radius(trial_centers)
                if trial_r > best_new_r:
                    best_new_r = trial_r
                    best_step = scale
            
            if best_step == 0.0:
                # Try a smaller random perturbation
                noise = np.random.randn(N, 3) * 0.001
                trial_centers = centers + noise
                trial_centers = np.clip(trial_centers, -0.5 + 1e-9, 0.5 - 1e-9)
                trial_r = get_min_radius(trial_centers)
                if trial_r > best_new_r:
                    best_new_r = trial_r
                    best_step = 1.0
                    grad_r = noise / (np.linalg.norm(noise) + 1e-10)
                else:
                    break
            
            centers = centers + best_step * grad_r
            centers = np.clip(centers, -0.5 + 1e-9, 0.5 - 1e-9)
            
            new_r = get_min_radius(centers)
            if new_r > best_local_r:
                best_local_r = new_r
                best_local_centers = centers.copy()
            
            # Check for convergence
            if abs(new_r - r) < tolerance:
                break
        
        return best_local_centers, best_local_r

    def optimize_soft(centers, max_steps=20000, beta_start=5.0, beta_end=2000.0, lr_start=0.008, lr_end=0.00005):
        """
        Optimizes the configuration by maximizing a soft-min surrogate of the radius.
        Uses log-sum-exp for numerical stability with improved annealing schedule.
        """
        centers = centers.copy()
        current_r = get_min_radius(centers)
        best_local_r = current_r
        best_local_centers = centers.copy()
        
        for step in range(max_steps):
            if time.time() - start_time > time_limit:
                break
            
            # Annealing parameters - use cosine annealing for smoother transition
            progress = step / max_steps
            beta = beta_end - (beta_end - beta_start) * 0.5 * (1 + np.cos(np.pi * progress))
            lr = lr_start * (lr_end / lr_start) ** progress
            
            # Compute gradient using the shared function
            grad = compute_gradients(centers, beta)
            
            # Update with adaptive step size
            grad_norm = np.linalg.norm(grad)
            if grad_norm > 1e-12:
                base_step = lr * max(current_r, 0.01) * 5.0
                best_step = base_step
                best_new_r = -np.inf
                for scale in [1.0, 0.5, 2.0]:
                    trial_centers = centers + base_step * scale * grad / grad_norm
                    trial_centers = np.clip(trial_centers, -0.5 + 1e-9, 0.5 - 1e-9)
                    trial_r = get_min_radius(trial_centers)
                    if trial_r > best_new_r:
                        best_new_r = trial_r
                        best_step = base_step * scale
                
                centers = centers + best_step * grad / grad_norm
                centers = np.clip(centers, -0.5 + 1e-9, 0.5 - 1e-9)
            
            # Evaluate
            new_r = get_min_radius(centers)
            current_r = new_r
            
            if new_r > best_local_r:
                best_local_r = new_r
                best_local_centers = centers.copy()
            
            # Add small noise to escape local minima periodically
            if step % 800 == 400 and step > 0:
                noise_scale = lr * 0.15 * max(current_r, 0.01)
                centers += np.random.randn(N, 3) * noise_scale
                centers = np.clip(centers, -0.5 + 1e-9, 0.5 - 1e-9)

        return best_local_centers, best_local_r

    def optimize_scipy(centers, max_time=20.0):
        """
        Use scipy's L-BFGS-B to polish the configuration.
        """
        centers_flat = centers.copy().flatten()
        start_opt = time.time()
        
        # Start with moderate beta, increase for final polish
        beta = 200.0
        
        def objective(x):
            c = x.reshape(N, 3)
            return -soft_min_value(c, beta)
        
        def gradient(x):
            c = x.reshape(N, 3)
            return compute_gradients(c, beta).flatten()
        
        try:
            # First pass with moderate beta
            res = minimize(objective, centers_flat, method='L-BFGS-B', jac=gradient,
                          bounds=[(-0.5+1e-9, 0.5-1e-9)] * (N*3),
                          options={'maxiter': 500, 'maxfun': 3000, 'ftol': 1e-14, 'gtol': 1e-10})
            
            if time.time() - start_opt > max_time * 0.6:
                new_centers = res.x.reshape(N, 3)
                new_r = get_min_radius(new_centers)
                if new_r > get_min_radius(centers):
                    return new_centers, new_r
                return centers, get_min_radius(centers)
            
            # Second pass with higher beta for precision
            beta = 500.0
            centers_flat = res.x
            
            def objective2(x):
                c = x.reshape(N, 3)
                return -soft_min_value(c, beta)
            
            def gradient2(x):
                c = x.reshape(N, 3)
                return compute_gradients(c, beta).flatten()
            
            res2 = minimize(objective2, centers_flat, method='L-BFGS-B', jac=gradient2,
                          bounds=[(-0.5+1e-9, 0.5-1e-9)] * (N*3),
                          options={'maxiter': 300, 'maxfun': 2000, 'ftol': 1e-16, 'gtol': 1e-12})
            
            new_centers = res2.x.reshape(N, 3)
            new_r = get_min_radius(new_centers)
            
            if new_r > get_min_radius(centers):
                return new_centers, new_r
            else:
                return centers, get_min_radius(centers)
                
        except Exception:
            return centers, get_min_radius(centers)

    def optimize_hard(centers, max_steps=5000, step_size=0.002):
        """
        Final polish using hard constraint logic to fix precision issues.
        """
        centers = centers.copy()
        current_r = get_min_radius(centers)
        best_local_r = current_r
        best_local_centers = centers.copy()
        
        step = step_size
        no_improve_count = 0
        
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
            
            if r_pair < r_wall:
                direction = (centers[i_min] - centers[j_min]) / (min_dist_val + 1e-10)
                push = 0.5 * step
                forces[i_min] += direction * push
                forces[j_min] -= direction * push
            else:
                if centers[i_wall, k_wall] > 0:
                    forces[i_wall, k_wall] -= 0.5 * step
                else:
                    forces[i_wall, k_wall] += 0.5 * step
            
            centers += forces
            centers = np.clip(centers, -0.5 + 1e-9, 0.5 - 1e-9)
            
            new_r = get_min_radius(centers)
            
            if new_r > best_local_r + 1e-15:
                best_local_r = new_r
                best_local_centers = centers.copy()
                no_improve_count = 0
            else:
                no_improve_count += 1
                step *= 0.9
                if step < 1e-10 or no_improve_count > 200:
                    break

        return best_local_centers, best_local_r

    # --- Main Loop ---
    
    # 1. Cubic Grid
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    init_grid = grid[:N]
    
    c1, r1 = optimize_soft(init_grid, max_steps=25000)
    if r1 > best_r:
        best_r = r1
        best_centers = c1
    
    # Active-set refinement
    if time.time() - start_time < time_limit - 15:
        c1_as, r1_as = optimize_active_set(c1, max_iterations=100)
        if r1_as > best_r:
            best_r = r1_as
            best_centers = c1_as
    
    # Scipy polish
    if time.time() - start_time < time_limit - 25:
        c1_scipy, r1_scipy = optimize_scipy(c1, max_time=18.0)
        if r1_scipy > best_r:
            best_r = r1_scipy
            best_centers = c1_scipy
    
    # Hard polish
    if time.time() - start_time < time_limit - 12:
        c1_pol, r1_pol = optimize_hard(c1, max_steps=4000, step_size=0.002)
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
    
    if time.time() - start_time < time_limit - 35:
        c2, r2 = optimize_soft(init_fcc, max_steps=25000)
        if r2 > best_r:
            best_r = r2
            best_centers = c2
            
        if time.time() - start_time < time_limit - 25:
            c2_as, r2_as = optimize_active_set(c2, max_iterations=100)
            if r2_as > best_r:
                best_r = r2_as
                best_centers = c2_as
        
        if time.time() - start_time < time_limit - 20:
            c2_scipy, r2_scipy = optimize_scipy(c2, max_time=18.0)
            if r2_scipy > best_r:
                best_r = r2_scipy
                best_centers = c2_scipy
                
        if time.time() - start_time < time_limit - 12:
            c2_pol, r2_pol = optimize_hard(c2, max_steps=4000, step_size=0.002)
            if r2_pol > best_r:
                best_r = r2_pol
                best_centers = c2_pol

    # 3. Random Initialization
    np.random.seed(42)
    init_random = np.random.uniform(-0.4, 0.4, (N, 3))
    
    if time.time() - start_time < time_limit - 35:
        c3, r3 = optimize_soft(init_random, max_steps=25000)
        if r3 > best_r:
            best_r = r3
            best_centers = c3
            
        if time.time() - start_time < time_limit - 25:
            c3_as, r3_as = optimize_active_set(c3, max_iterations=100)
            if r3_as > best_r:
                best_r = r3_as
                best_centers = c3_as
        
        if time.time() - start_time < time_limit - 20:
            c3_scipy, r3_scipy = optimize_scipy(c3, max_time=18.0)
            if r3_scipy > best_r:
                best_r = r3_scipy
                best_centers = c3_scipy
                
        if time.time() - start_time < time_limit - 12:
            c3_pol, r3_pol = optimize_hard(c3, max_steps=4000, step_size=0.002)
            if r3_pol > best_r:
                best_r = r3_pol
                best_centers = c3_pol

    # 4. Final Polish on the best found
    if best_centers is not None and time.time() - start_time < time_limit - 15:
        c_best_as, r_best_as = optimize_active_set(best_centers, max_iterations=150)
        if r_best_as > best_r:
            best_r = r_best_as
            best_centers = c_best_as
        
        if time.time() - start_time < time_limit - 10:
            c_best_scipy, r_best_scipy = optimize_scipy(best_centers, max_time=12.0)
            if r_best_scipy > best_r:
                best_r = r_best_scipy
                best_centers = c_best_scipy
            
            if time.time() - start_time < time_limit - 3:
                c_best, r_best = optimize_hard(best_centers, max_steps=6000, step_size=0.001)
                if r_best > best_r:
                    best_r = r_best
                    best_centers = c_best

    if best_centers is None:
        return init_grid

    return best_centers
