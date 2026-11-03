"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    time_limit = 145.0  # Leave 5s buffer for 150s limit
    
    def min_radius(c):
        """Compute the minimum radius for a given set of centers."""
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        d = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(d, np.inf)
        min_pair_dist = np.min(d)
        
        dist_to_walls = np.min(np.minimum(c + 0.5, 0.5 - c), axis=1)
        min_wall_dist = np.min(dist_to_walls)
        
        return min(min_pair_dist / 2.0, min_wall_dist)
    
    def soft_min_radius(c, beta):
        """
        Soft-min surrogate using log-sum-exp for smooth optimization.
        Returns a smooth approximation of the minimum radius.
        """
        # Pairwise distances
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        d_sq = np.sum(diff * diff, axis=2)
        np.fill_diagonal(d_sq, np.inf)
        
        # Wall distances
        dist_to_walls = np.min(np.minimum(c + 0.5, 0.5 - c), axis=1)
        
        # Convert to squared distances for consistency
        pair_dists_sq = d_sq / 4.0  # (d/2)^2
        wall_dists_sq = dist_to_walls ** 2
        
        # Combine all constraints into one array
        all_dists_sq = np.concatenate([pair_dists_sq.flatten(), wall_dists_sq])
        
        # Log-sum-exp for stable computation
        max_val = np.max(all_dists_sq)
        if max_val < 1e-15:
            return 0.0
        
        # Use soft min: -log(sum(exp(-beta * x))) / beta
        # This approximates min(x) as beta -> infinity
        exp_vals = np.exp(-beta * (all_dists_sq - max_val))
        sum_exp = np.sum(exp_vals)
        
        if sum_exp < 1e-300:
            return np.sqrt(max_val)
        
        log_sum_exp = -np.log(sum_exp) / beta
        soft_min_sq = max_val + log_sum_exp
        
        if soft_min_sq < 0:
            soft_min_sq = 0.0
        
        return np.sqrt(soft_min_sq)
    
    def soft_min_radius_grad(c, beta):
        """
        Gradient of the soft-min radius surrogate.
        """
        # Pairwise distances
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        d_sq = np.sum(diff * diff, axis=2)
        np.fill_diagonal(d_sq, np.inf)
        
        # Wall distances
        dist_to_walls = np.min(np.minimum(c + 0.5, 0.5 - c), axis=1)
        
        # Convert to squared distances
        pair_dists_sq = d_sq / 4.0
        wall_dists_sq = dist_to_walls ** 2
        
        all_dists_sq = np.concatenate([pair_dists_sq.flatten(), wall_dists_sq])
        
        max_val = np.max(all_dists_sq)
        if max_val < 1e-15:
            return np.zeros_like(c)
        
        exp_vals = np.exp(-beta * (all_dists_sq - max_val))
        sum_exp = np.sum(exp_vals)
        
        if sum_exp < 1e-300:
            return np.zeros_like(c)
        
        # Weights for soft min
        weights = exp_vals / sum_exp
        
        # Gradient of soft_min_sq = max_val + log(sum(exp(-beta*(x - max_val)))) / (-beta)
        # d(soft_min_sq)/dx_i = -weights_i (approximately, when max_val is constant)
        
        # For the actual gradient of soft_min = sqrt(soft_min_sq):
        # d(soft_min)/dx_i = (1/(2*soft_min)) * d(soft_min_sq)/dx_i
        
        soft_min_sq = max_val - np.log(sum_exp) / beta
        if soft_min_sq < 1e-15:
            return np.zeros_like(c)
        
        soft_min = np.sqrt(soft_min_sq)
        
        # Gradient of soft_min_sq with respect to c
        grad_sq = np.zeros_like(c)
        
        # Pairwise contribution
        # d(pair_dists_sq[i,j])/dc_i = (c_i - c_j) / 2
        # d(pair_dists_sq[i,j])/dc_j = (c_j - c_i) / 2
        n_pairs = N * (N - 1) // 2
        pair_weights = weights[:n_pairs]
        
        # Create index array for pairs
        idx_i = np.array([i for i in range(N) for j in range(i + 1, N)])
        idx_j = np.array([j for i in range(N) for j in range(i + 1, N)])
        
        for k in range(n_pairs):
            i, j = idx_i[k], idx_j[k]
            w = pair_weights[k]
            if w > 1e-10:
                vec = c[i] - c[j]
                grad_sq[i] -= vec * w / 2.0
                grad_sq[j] += vec * w / 2.0
        
        # Wall contribution
        wall_weights = weights[n_pairs:]
        for i in range(N):
            w = wall_weights[i]
            if w > 1e-10:
                for k in range(3):
                    if c[i, k] > 0:
                        grad_sq[i, k] += w  # d((0.5-c)^2)/dc = -2(0.5-c), but we want to move away from wall
                    else:
                        grad_sq[i, k] -= w  # d((c+0.5)^2)/dc = 2(c+0.5)
        
        # Chain rule: d(soft_min)/dc = (1/(2*soft_min)) * d(soft_min_sq)/dc
        grad = grad_sq / (2.0 * soft_min)
        
        return grad
    
    def spectral_init(n_points, seed=0):
        """Generate diverse initial configurations using spectral methods."""
        rng = np.random.RandomState(seed)
        
        # Start with a random configuration
        centers = rng.uniform(-0.4, 0.4, size=(n_points, 3))
        
        # Perform some relaxation iterations
        for _ in range(50):
            # Compute forces
            forces = np.zeros_like(centers)
            diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
            d = np.linalg.norm(diff, axis=2)
            np.fill_diagonal(d, np.inf)
            
            min_d = np.min(d)
            target_d = min_d * 1.2
            
            for i in range(n_points):
                for j in range(i + 1, n_points):
                    if d[i, j] < target_d:
                        vec = centers[i] - centers[j]
                        norm = d[i, j]
                        if norm > 1e-9:
                            force = (target_d - d[i, j]) / norm
                            forces[i] += vec * force
                            forces[j] -= vec * force
            
            # Wall repulsion
            for i in range(n_points):
                for k in range(3):
                    if centers[i, k] > 0.3:
                        forces[i, k] -= (centers[i, k] - 0.3) * 2.0
                    elif centers[i, k] < -0.3:
                        forces[i, k] -= (centers[i, k] + 0.3) * 2.0
            
            centers += forces * 0.1
            centers = np.clip(centers, -0.45, 0.45)
        
        return centers
    
    # Generate multiple diverse initial configurations
    n_starts = 8
    initial_configs = []
    for i in range(n_starts):
        if i == 0:
            # Grid initialization
            m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
            ticks = (np.arange(m) + 0.5) / m - 0.5
            grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
            initial_configs.append(grid[:N].copy())
        else:
            initial_configs.append(spectral_init(N, seed=i * 100 + 42))
    
    best_centers = None
    best_r = -1.0
    
    # Time budget allocation
    time_per_start = (time_limit - 5.0) / n_starts
    
    for start_idx, init_centers in enumerate(initial_configs):
        if time.time() - start_time > time_limit - 5.0:
            break
        
        centers = init_centers.copy()
        current_r = min_radius(centers)
        
        # Annealed beta schedule for L-BFGS-B optimization
        beta_schedule = [10, 50, 100, 200, 500, 1000]
        
        for beta in beta_schedule:
            if time.time() - start_time > time_limit - 5.0:
                break
            
            # Use L-BFGS-B with the soft-min surrogate
            def objective(x):
                c = x.reshape(-1, 3)
                return -soft_min_radius(c, beta)
            
            def gradient(x):
                c = x.reshape(-1, 3)
                return -soft_min_radius_grad(c, beta).flatten()
            
            # Bounds
            bounds = [(-0.5 + 1e-9, 0.5 - 1e-9)] * (N * 3)
            
            try:
                result = minimize(
                    objective,
                    centers.flatten(),
                    jac=gradient,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                centers = result.x.reshape(-1, 3)
            except Exception:
                pass
            
            current_r = min_radius(centers)
            
            if current_r > best_r:
                best_r = current_r
                best_centers = centers.copy()
        
        # Final polish with very high beta
        if time.time() - start_time < time_limit - 3.0:
            beta_final = 10000
            
            def objective_final(x):
                c = x.reshape(-1, 3)
                return -soft_min_radius(c, beta_final)
            
            def gradient_final(x):
                c = x.reshape(-1, 3)
                return -soft_min_radius_grad(c, beta_final).flatten()
            
            try:
                result = minimize(
                    objective_final,
                    centers.flatten(),
                    jac=gradient_final,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-12}
                )
                centers = result.x.reshape(-1, 3)
            except Exception:
                pass
            
            current_r = min_radius(centers)
            if current_r > best_r:
                best_r = current_r
                best_centers = centers.copy()
    
    # Ensure we have a valid result
    if best_centers is None:
        m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        best_centers = grid[:N].copy()
    
    return best_centers
# EVOLVE-BLOCK-END
