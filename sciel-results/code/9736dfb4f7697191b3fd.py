import numpy as np
import time

N = 100

def construct():
    start_time = time.time()
    time_limit = 140.0  # Leave 10s margin for the 150s limit

    best_r = 0.0
    best_centers = None

    def get_min_radius(centers):
        # Compute pairwise distances
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dist, np.inf)
        min_pair_dist = np.min(dist)
        
        # Wall distances
        wall_dists = 0.5 - np.abs(centers)
        min_wall_dist = np.min(wall_dists)
        
        return min(min_pair_dist / 2.0, min_wall_dist)

    def optimize(centers, max_steps=30000, initial_step=0.01, temperature=1.0):
        """
        Force-directed relaxation with simulated annealing-like perturbation.
        """
        centers = centers.copy()
        current_r = get_min_radius(centers)
        best_local_r = current_r
        best_local_centers = centers.copy()
        
        step = initial_step
        no_improve_count = 0
        max_no_improve = 1500
        
        # Annealing parameters
        cooling_rate = 0.9995
        min_step = 0.0005
        
        for i in range(max_steps):
            if time.time() - start_time > time_limit:
                break
            
            # Compute all pairwise distances
            diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
            dist = np.linalg.norm(diff, axis=2)
            np.fill_diagonal(dist, np.inf)
            
            # Compute repulsion forces from all neighbors
            # Soft repulsion: f = strength / (dist^2 + epsilon)
            inv_dist_sq = 1.0 / (dist**2 + 1e-10)
            inv_dist_sq = np.where(np.eye(N, dtype=bool), 0.0, inv_dist_sq)
            inv_dist_sq = np.minimum(inv_dist_sq, 100.0)  # Cap to prevent explosion
            
            # Vectorized force calculation
            row_sums = np.sum(inv_dist_sq, axis=1)
            weighted_centers = inv_dist_sq @ centers
            forces = centers * row_sums[:, np.newaxis] - weighted_centers
            
            # Wall repulsion: push points away from walls
            # Use a smooth potential that activates near walls
            for k in range(3):
                dist_to_pos_wall = 0.5 - centers[:, k]
                dist_to_neg_wall = 0.5 + centers[:, k]
                
                # Positive wall repulsion
                mask_pos = dist_to_pos_wall < 0.15
                if np.any(mask_pos):
                    f = 0.5 / (dist_to_pos_wall[mask_pos]**2 + 1e-10)
                    f = np.minimum(f, 50.0)
                    forces[mask_pos, k] -= f
                
                # Negative wall repulsion
                mask_neg = dist_to_neg_wall < 0.15
                if np.any(mask_neg):
                    f = 0.5 / (dist_to_neg_wall[mask_neg]**2 + 1e-10)
                    f = np.minimum(f, 50.0)
                    forces[mask_neg, k] += f
            
            # Normalize forces to prevent large steps
            force_magnitudes = np.linalg.norm(forces, axis=1)
            max_force = np.max(force_magnitudes) if np.max(force_magnitudes) > 0 else 1.0
            if max_force > 1.0:
                forces /= max_force
            
            # Update positions
            centers += step * forces
            
            # Clamp to box
            centers = np.clip(centers, -0.4999, 0.4999)
            
            # Compute new radius
            new_r = get_min_radius(centers)
            
            # Adaptive step size and annealing
            if new_r > current_r:
                # Improvement: increase step slightly, reset no_improve
                step = min(step * 1.05, 0.02)
                no_improve_count = 0
                if new_r > best_local_r:
                    best_local_r = new_r
                    best_local_centers = centers.copy()
            else:
                # No improvement: decrease step, increase no_improve
                step = max(step * 0.95, min_step)
                no_improve_count += 1
                
                # If stuck for a long time, apply a random perturbation
                # The magnitude of the perturbation is scaled by the current step size
                if no_improve_count > max_no_improve:
                    n_perturb = max(1, N // 10)
                    perturbed = np.random.choice(N, n_perturb, replace=False)
                    # Perturbation magnitude scales with step size to allow larger jumps when step is large
                    perturbation_magnitude = step * 2.0
                    centers[perturbed] += np.random.randn(n_perturb, 3) * perturbation_magnitude
                    centers = np.clip(centers, -0.4999, 0.4999)
                    no_improve_count = 0
                    # Reset step to a moderate value after perturbation
                    step = min(0.01, step * 2.0)
            
            current_r = new_r
            
            # Global cooling: slowly decrease the maximum allowed step size
            # This ensures we eventually fine-tune
            # temperature is not directly used here but controls the cooling of step size upper bound
            # We can adjust the upper bound of step based on iteration count
            max_step_allowed = initial_step * (0.5 ** (i / (max_steps / 3.0)))
            if step > max_step_allowed:
                step = max_step_allowed

        return best_local_centers, best_local_r

    # Initial Configurations
    
    # 1. Cubic Grid (Seed)
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    init_grid = grid[:N]
    
    # 2. Jittered Grid (Better exploration than pure random)
    np.random.seed(42)
    init_jittered = init_grid + np.random.uniform(-0.05, 0.05, (N, 3))
    init_jittered = np.clip(init_jittered, -0.49, 0.49)
    
    # 3. FCC-like Initialization
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
    
    # 4. Random Initialization
    np.random.seed(123)
    init_random = np.random.uniform(-0.45, 0.45, (N, 3))
    
    # Run optimization on each initial configuration
    # We allocate time based on the number of starts
    # 4 starts, so roughly 35s each
    starts = [
        ("grid", init_grid),
        ("jittered", init_jittered),
        ("fcc", init_fcc),
        ("random", init_random)
    ]
    
    for name, init_c in starts:
        if time.time() - start_time > time_limit - 10:
            break
        c, r = optimize(init_c, max_steps=30000, initial_step=0.01)
        if r > best_r:
            best_r = r
            best_centers = c

    # If time remains, further optimize the best configuration found
    remaining_time = time_limit - (time.time() - start_time)
    if remaining_time > 15 and best_centers is not None:
        # Fine tuning pass with smaller initial step
        c_best, r_best = optimize(best_centers, max_steps=20000, initial_step=0.005)
        if r_best > best_r:
            best_r = r_best
            best_centers = c_best

    if best_centers is None:
        return init_grid

    return best_centers
