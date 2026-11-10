import numpy as np
import time
from scipy.spatial.distance import cdist

N = 100
DEADLINE = 130.0  # Leave margin before 150s kill

def construct():
    start_time = time.time()
    
    best_r = 0.0
    best_centers = None
    
    def remaining_time():
        return DEADLINE - (time.time() - start_time)
    
    def get_min_radius(centers):
        # Compute pairwise distances
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff * diff, axis=2))
        np.fill_diagonal(dist, np.inf)
        min_pair_dist = np.min(dist)
        
        # Wall distances
        wall_dists = 0.5 - np.abs(centers)
        min_wall_dist = np.min(wall_dists)
        
        return min(min_pair_dist / 2.0, min_wall_dist)
    
    def soft_min_radius(centers, beta=50.0):
        """Smooth surrogate for the minimum radius using log-sum-exp."""
        # Pairwise distances
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dist_sq = np.sum(diff * diff, axis=2)
        np.fill_diagonal(dist_sq, np.inf)
        
        # Wall distances
        wall_dists = 0.5 - np.abs(centers)
        
        # All constraints: pair distances/2 and wall distances
        # For log-sum-exp, we want to minimize the "soft min"
        # soft_min(x) = -logsumexp(-beta*x) / beta
        # This approximates min(x) as beta -> infinity
        
        # Pair distances (divided by 2)
        pair_dists = np.sqrt(dist_sq) / 2.0
        np.fill_diagonal(pair_dists, np.inf)
        
        # Flatten all constraint values
        all_vals = np.concatenate([pair_dists.ravel(), wall_dists.ravel()])
        all_vals = all_vals[np.isfinite(all_vals)]
        
        if len(all_vals) == 0:
            return 0.0
        
        # Log-sum-exp trick for numerical stability
        max_val = np.max(all_vals)
        exp_vals = np.exp(-beta * (all_vals - max_val))
        logsumexp = np.log(np.sum(exp_vals)) + (-beta * max_val)
        
        soft_min = -logsumexp / beta
        
        return soft_min
    
    def compute_forces(centers, repulsion_strength=1.0, wall_strength=2.0):
        """Compute repulsion forces with smooth wall repulsion."""
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dist_sq = np.sum(diff * diff, axis=2)
        np.fill_diagonal(dist_sq, np.inf)
        
        # Pair repulsion: f = strength / (dist^2 + epsilon)
        inv_dist_sq = 1.0 / (dist_sq + 1e-10)
        inv_dist_sq = np.where(np.eye(N, dtype=bool), 0.0, inv_dist_sq)
        inv_dist_sq = np.minimum(inv_dist_sq, 50.0)  # Cap to prevent explosion
        
        row_sums = np.sum(inv_dist_sq, axis=1)
        weighted_centers = inv_dist_sq @ centers
        
        forces = centers * row_sums[:, np.newaxis] - weighted_centers
        forces *= repulsion_strength
        
        # Smooth wall repulsion: always active, stronger when close
        for k in range(3):
            dist_to_pos_wall = 0.5 - centers[:, k]
            dist_to_neg_wall = 0.5 + centers[:, k]
            
            # Smooth repulsion: strength / (dist + epsilon)^2, capped
            pos_force = wall_strength / (dist_to_pos_wall + 0.01)**2
            neg_force = wall_strength / (dist_to_neg_wall + 0.01)**2
            
            pos_force = np.minimum(pos_force, 50.0)
            neg_force = np.minimum(neg_force, 50.0)
            
            forces[:, k] += -pos_force + neg_force
        
        return forces
    
    def optimize(centers, max_steps=30000, step_size=0.005, repulsion_strength=1.0, wall_strength=2.0):
        """Force-directed relaxation with smooth forces."""
        centers = centers.copy()
        current_r = get_min_radius(centers)
        best_local_r = current_r
        best_local_centers = centers.copy()
        
        # Adaptive step size
        step = step_size
        no_improve_count = 0
        max_no_improve = 1500
        
        for i in range(max_steps):
            if remaining_time() < 5.0:
                break
            
            # Compute forces
            forces = compute_forces(centers, repulsion_strength, wall_strength)
            
            # Normalize forces
            force_magnitudes = np.linalg.norm(forces, axis=1)
            max_force = np.max(force_magnitudes) if np.max(force_magnitudes) > 0 else 1.0
            if max_force > 1.0:
                forces /= max_force
            
            # Update positions
            centers += step * forces
            
            # Clamp to box
            centers = np.clip(centers, -0.499, 0.499)
            
            # Compute new radius
            new_r = get_min_radius(centers)
            
            # Adaptive step size
            if new_r > current_r + 1e-12:
                step = min(step * 1.05, 0.015)
                no_improve_count = 0
                if new_r > best_local_r:
                    best_local_r = new_r
                    best_local_centers = centers.copy()
            else:
                step = max(step * 0.95, 0.002)
                no_improve_count += 1
            
            current_r = new_r
            
            # If stuck, add random perturbation
            if no_improve_count > max_no_improve:
                n_perturb = max(1, N // 8)
                perturbed = np.random.choice(N, n_perturb, replace=False)
                centers[perturbed] += np.random.randn(n_perturb, 3) * 0.015
                centers = np.clip(centers, -0.499, 0.499)
                no_improve_count = 0
                step = 0.005
        
        return best_local_centers, best_local_r
    
    def polish(centers, max_steps=10000, step_size=0.002):
        """High-precision polish with very small steps."""
        centers = centers.copy()
        current_r = get_min_radius(centers)
        best_local_r = current_r
        best_local_centers = centers.copy()
        
        step = step_size
        
        for i in range(max_steps):
            if remaining_time() < 3.0:
                break
            
            forces = compute_forces(centers, 0.5, 1.0)
            
            force_magnitudes = np.linalg.norm(forces, axis=1)
            max_force = np.max(force_magnitudes) if np.max(force_magnitudes) > 0 else 1.0
            if max_force > 1.0:
                forces /= max_force
            
            centers += step * forces
            centers = np.clip(centers, -0.499, 0.499)
            
            new_r = get_min_radius(centers)
            
            if new_r > best_local_r:
                best_local_r = new_r
                best_local_centers = centers.copy()
            
            current_r = new_r
        
        return best_local_centers, best_local_r

    # Initial Configurations
    
    # 1. Cubic Grid (Seed)
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    init_grid = grid[:N]
    
    c1, r1 = optimize(init_grid, max_steps=20000)
    if r1 > best_r:
        best_r = r1
        best_centers = c1

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
    
    c2, r2 = optimize(init_fcc, max_steps=20000)
    if r2 > best_r:
        best_r = r2
        best_centers = c2

    # 3. Random Initialization
    np.random.seed(42)
    init_random = np.random.uniform(-0.4, 0.4, (N, 3))
    c3, r3 = optimize(init_random, max_steps=20000)
    if r3 > best_r:
        best_r = r3
        best_centers = c3

    # 4. If time remains, do a high-precision polish on the best configuration
    if remaining_time() > 10 and best_centers is not None:
        c_best, r_best = polish(best_centers, max_steps=15000, step_size=0.002)
        if r_best > best_r:
            best_r = r_best
            best_centers = c_best

    # 5. One more polish if time permits
    if remaining_time() > 5 and best_centers is not None:
        c_best, r_best = polish(best_centers, max_steps=8000, step_size=0.001)
        if r_best > best_r:
            best_r = r_best
            best_centers = c_best

    if best_centers is None:
        return init_grid

    return best_centers
