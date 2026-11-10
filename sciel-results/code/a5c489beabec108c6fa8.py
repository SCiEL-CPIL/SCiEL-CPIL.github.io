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

    def optimize_soft(centers, max_steps=20000, beta_start=5.0, beta_end=2000.0, lr_start=0.008, lr_end=0.00005):
        """
        Optimizes the configuration by maximizing a soft-min surrogate of the radius.
        Uses log-sum-exp for numerical stability with improved annealing schedule.
        """
        centers = centers.copy()
        current_r = get_min_radius(centers)
        best_local_r = current_r
        best_local_centers = centers.copy()
        
        mask = np.triu(np.ones((N, N), dtype=bool), k=1)
        
        for step in range(max_steps):
            if time.time() - start_time > time_limit:
                break
            
            # Annealing parameters - use cosine annealing for smoother transition
            progress = step / max_steps
            # Cosine annealing: starts high, ends low
            beta = beta_end - (beta_end - beta_start) * 0.5 * (1 + np.cos(np.pi * progress))
            lr = lr_start * (lr_end / lr_start) ** progress
            
            # 1. Compute all pairwise distances and wall distances
            diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
            dist_mat = np.linalg.norm(diff, axis=2)
            np.fill_diagonal(dist_mat, 1e10)
            
            # Wall distances
            wall_dists = np.empty((N, 6))
            wall_dists[:, 0] = centers[:, 0] + 0.5
            wall_dists[:, 1] = 0.5 - centers[:, 0]
            wall_dists[:, 2] = centers[:, 1] + 0.5
            wall_dists[:, 3] = 0.5 - centers[:, 1]
            wall_dists[:, 4] = centers[:, 2] + 0.5
            wall_dists[:, 5] = 0.5 - centers[:, 2]
            
            # Effective distances: r = 0.5 * min(d_eff)
            d_pair_eff = dist_mat
            d_wall_eff = 2.0 * wall_dists
            
            pair_dists = d_pair_eff[mask]
            wall_dists_flat = d_wall_eff.flatten()
            
            # 2. Compute weights using log-sum-exp for stability
            all_dists = np.concatenate([pair_dists, wall_dists_flat])
            max_d = np.max(all_dists)
            
            exp_pairs = np.exp(-beta * (pair_dists - max_d))
            exp_walls = np.exp(-beta * (wall_dists_flat - max_d))
            
            sum_pairs = np.sum(exp_pairs)
            sum_walls = np.sum(exp_walls)
            total_sum = sum_pairs + sum_walls
            
            if total_sum < 1e-300:
                break
                
            w_pairs = exp_pairs / total_sum
            w_walls = exp_walls / total_sum
            
            # 3. Compute Gradient
            grad = np.zeros_like(centers)
            
            # Pair contributions
            W_pair_mat = np.zeros((N, N))
            W_pair_mat[mask] = w_pairs
            
            U_pair = np.zeros((N, N))
            u_flat = w_pairs / (pair_dists + 1e-10)
            U_pair[mask] = u_flat
            
            row_sums_U = np.sum(U_pair, axis=1)
            weighted_c = U_pair @ centers
            
            grad_pairs = centers * row_sums_U[:, np.newaxis] - weighted_c
            
            # Wall contributions
            W_wall_mat = w_walls.reshape(N, 6)
            
            grad_walls = np.zeros_like(centers)
            grad_walls[:, 0] = 2.0 * W_wall_mat[:, 0] - 2.0 * W_wall_mat[:, 1]
            grad_walls[:, 1] = 2.0 * W_wall_mat[:, 2] - 2.0 * W_wall_mat[:, 3]
            grad_walls[:, 2] = 2.0 * W_wall_mat[:, 4] - 2.0 * W_wall_mat[:, 5]
            
            grad = grad_pairs + grad_walls
            
            # 4. Update with adaptive step size using line search
            grad_norm = np.linalg.norm(grad)
            if grad_norm > 1e-12:
                # Adaptive step size based on gradient magnitude and current radius
                base_step = lr * max(current_r, 0.01) * 5.0
                # Try a few step sizes to find one that improves
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
            
            # 5. Evaluate
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
        Maximize the soft-min objective with high precision using adaptive beta.
        """
        centers = centers.copy().flatten()
        start_opt = time.time()
        
        # Start with moderate beta, increase for final polish
        beta = 200.0
        
        def objective(x):
            c = x.reshape(N, 3)
            diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
            dist = np.linalg.norm(diff, axis=2)
            np.fill_diagonal(dist, 1e10)
            
            wall_dists = 0.5 - np.abs(c)
            
            d_pair = dist[np.triu_indices(N, k=1)]
            d_wall = 2.0 * wall_dists.flatten()
            
            all_d = np.concatenate([d_pair, d_wall])
            
            max_d = np.max(all_d)
            log_sum = np.log(np.sum(np.exp(-beta * (all_d - max_d))))
            soft_min = -log_sum / beta + max_d
            
            return -soft_min
        
        def gradient(x):
            c = x.reshape(N, 3)
            diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
            dist = np.linalg.norm(diff, axis=2)
            np.fill_diagonal(dist, 1e10)
            
            wall_dists = 0.5 - np.abs(c)
            d_wall_eff = 2.0 * wall_dists
            
            mask = np.triu(np.ones((N, N), dtype=bool), k=1)
            pair_dists = dist[mask]
            wall_flat = d_wall_eff.flatten()
            
            all_d = np.concatenate([pair_dists, wall_flat])
            max_d = np.max(all_d)
            
            exp_pairs = np.exp(-beta * (pair_dists - max_d))
            exp_walls = np.exp(-beta * (wall_flat - max_d))
            
            sum_pairs = np.sum(exp_pairs)
            sum_walls = np.sum(exp_walls)
            total = sum_pairs + sum_walls
            
            if total < 1e-300:
                return np.zeros_like(x)
            
            w_pairs = exp_pairs / total
            w_walls = exp_walls / total
            
            grad = np.zeros_like(c)
            
            W_pair_mat = np.zeros((N, N))
            W_pair_mat[mask] = w_pairs
            U_pair = np.zeros((N, N))
            U_pair[mask] = w_pairs / (pair_dists + 1e-10)
            
            row_sums = np.sum(U_pair, axis=1)
            weighted_c = U_pair @ c
            grad_pairs = c * row_sums[:, np.newaxis] - weighted_c
            
            W_wall_mat = w_walls.reshape(N, 6)
            grad_walls = np.zeros_like(c)
            grad_walls[:, 0] = 2.0 * W_wall_mat[:, 0] - 2.0 * W_wall_mat[:, 1]
            grad_walls[:, 1] = 2.0 * W_wall_mat[:, 2] - 2.0 * W_wall_mat[:, 3]
            grad_walls[:, 2] = 2.0 * W_wall_mat[:, 4] - 2.0 * W_wall_mat[:, 5]
            
            grad = grad_pairs + grad_walls
            return (-beta * grad).flatten()
        
        try:
            # First pass with moderate beta
            res = minimize(objective, centers, method='L-BFGS-B', jac=gradient,
                          bounds=[(-0.5+1e-9, 0.5-1e-9)] * (N*3),
                          options={'maxiter': 500, 'maxfun': 3000, 'ftol': 1e-14, 'gtol': 1e-10})
            
            if time.time() - start_opt > max_time * 0.6:
                new_centers = res.x.reshape(N, 3)
                new_r = get_min_radius(new_centers)
                if new_r > get_min_radius(centers.reshape(N, 3)):
                    return new_centers, new_r
                return centers.reshape(N, 3), get_min_radius(centers.reshape(N, 3))
            
            # Second pass with higher beta for precision
            beta = 500.0
            centers = res.x
            
            def objective2(x):
                c = x.reshape(N, 3)
                diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
                dist = np.linalg.norm(diff, axis=2)
                np.fill_diagonal(dist, 1e10)
                
                wall_dists = 0.5 - np.abs(c)
                
                d_pair = dist[np.triu_indices(N, k=1)]
                d_wall = 2.0 * wall_dists.flatten()
                
                all_d = np.concatenate([d_pair, d_wall])
                
                max_d = np.max(all_d)
                log_sum = np.log(np.sum(np.exp(-beta * (all_d - max_d))))
                soft_min = -log_sum / beta + max_d
                
                return -soft_min
            
            def gradient2(x):
                c = x.reshape(N, 3)
                diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
                dist = np.linalg.norm(diff, axis=2)
                np.fill_diagonal(dist, 1e10)
                
                wall_dists = 0.5 - np.abs(c)
                d_wall_eff = 2.0 * wall_dists
                
                mask = np.triu(np.ones((N, N), dtype=bool), k=1)
                pair_dists = dist[mask]
                wall_flat = d_wall_eff.flatten()
                
                all_d = np.concatenate([pair_dists, wall_flat])
                max_d = np.max(all_d)
                
                exp_pairs = np.exp(-beta * (pair_dists - max_d))
                exp_walls = np.exp(-beta * (wall_flat - max_d))
                
                sum_pairs = np.sum(exp_pairs)
                sum_walls = np.sum(exp_walls)
                total = sum_pairs + sum_walls
                
                if total < 1e-300:
                    return np.zeros_like(x)
                
                w_pairs = exp_pairs / total
                w_walls = exp_walls / total
                
                grad = np.zeros_like(c)
                
                W_pair_mat = np.zeros((N, N))
                W_pair_mat[mask] = w_pairs
                U_pair = np.zeros((N, N))
                U_pair[mask] = w_pairs / (pair_dists + 1e-10)
                
                row_sums = np.sum(U_pair, axis=1)
                weighted_c = U_pair @ c
                grad_pairs = c * row_sums[:, np.newaxis] - weighted_c
                
                W_wall_mat = w_walls.reshape(N, 6)
                grad_walls = np.zeros_like(c)
                grad_walls[:, 0] = 2.0 * W_wall_mat[:, 0] - 2.0 * W_wall_mat[:, 1]
                grad_walls[:, 1] = 2.0 * W_wall_mat[:, 2] - 2.0 * W_wall_mat[:, 3]
                grad_walls[:, 2] = 2.0 * W_wall_mat[:, 4] - 2.0 * W_wall_mat[:, 5]
                
                grad = grad_pairs + grad_walls
                return (-beta * grad).flatten()
            
            res2 = minimize(objective2, centers, method='L-BFGS-B', jac=gradient2,
                          bounds=[(-0.5+1e-9, 0.5-1e-9)] * (N*3),
                          options={'maxiter': 300, 'maxfun': 2000, 'ftol': 1e-16, 'gtol': 1e-12})
            
            new_centers = res2.x.reshape(N, 3)
            new_r = get_min_radius(new_centers)
            
            if new_r > get_min_radius(centers.reshape(N, 3)):
                return new_centers, new_r
            else:
                return centers.reshape(N, 3), get_min_radius(centers.reshape(N, 3))
                
        except Exception:
            return centers.reshape(N, 3), get_min_radius(centers.reshape(N, 3))

    def optimize_hard(centers, max_steps=5000, step_size=0.002):
        """
        Final polish using hard constraint logic to fix precision issues
        or fine-tune the tightest constraints. Uses adaptive step size.
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
            
            # Apply force to the bottleneck with adaptive magnitude
            if r_pair < r_wall:
                # Push pair apart
                direction = (centers[i_min] - centers[j_min]) / (min_dist_val + 1e-10)
                push = 0.5 * step
                forces[i_min] += direction * push
                forces[j_min] -= direction * push
            else:
                # Push away from wall
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
    if best_centers is not None and time.time() - start_time < time_limit - 8:
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
