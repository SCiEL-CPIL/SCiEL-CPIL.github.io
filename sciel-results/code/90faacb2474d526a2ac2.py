import numpy as np
import time
from scipy.optimize import minimize

N = 100

def _compute_radius(centers):
    """Compute the common radius for a set of centers."""
    wall_clear = 0.5 - np.max(np.abs(centers), axis=1)
    min_wall = np.min(wall_clear)
    
    if len(centers) > 1:
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt((diff ** 2).sum(axis=2))
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists) / 2.0
    else:
        min_pair = min_wall
    
    return min(min_pair, min_wall)

def _force_directed_relax(centers, R, n_steps=200, dt=0.001, temp=0.01):
    """Force-directed relaxation with repulsive forces between spheres and walls."""
    centers = centers.copy()
    rng = np.random.RandomState(42)
    
    for step in range(n_steps):
        # Decay temperature for simulated annealing effect
        current_temp = temp * (1 - step / n_steps)
        
        # Compute forces
        forces = np.zeros_like(centers)
        
        # Pairwise repulsive forces
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt((diff ** 2).sum(axis=2))
        np.fill_diagonal(dists, 1e-10)
        
        # Repulsion strength: stronger when spheres are closer than 2R
        for i in range(N):
            for j in range(i + 1, N):
                d = dists[i, j]
                if d < 2 * R:
                    # Strong repulsion
                    strength = 10.0 / (d + 1e-10)
                    direction = (centers[i] - centers[j]) / d
                    forces[i] += strength * direction
                    forces[j] -= strength * direction
                elif d < 3 * R:
                    # Moderate repulsion to encourage spacing
                    strength = 0.5 / (d + 1e-10)
                    direction = (centers[i] - centers[j]) / d
                    forces[i] += strength * direction
                    forces[j] -= strength * direction
        
        # Wall repulsion: push spheres away from walls
        for i in range(N):
            for k in range(3):
                coord = centers[i, k]
                if coord > 0.5 - R:
                    # Too close to +wall
                    excess = coord - (0.5 - R)
                    if excess > 0:
                        forces[i, k] -= 5.0 * excess
                elif coord < -(0.5 - R):
                    # Too close to -wall
                    excess = -(0.5 - R) - coord
                    if excess > 0:
                        forces[i, k] += 5.0 * excess
        
        # Add small random perturbation to escape local minima
        forces += rng.normal(0, current_temp, size=centers.shape)
        
        # Update positions with velocity damping
        centers += dt * forces
        centers *= 0.99  # Damping
        
        # Clip to cube bounds
        centers = np.clip(centers, -0.499, 0.499)
    
    return centers

def _local_refine(centers, R, max_iter=500):
    """Local refinement using gradient-based optimization with adaptive beta."""
    x = centers.flatten()
    bounds = [(-0.5, 0.5)] * (N * 3)
    
    best_r = _compute_radius(centers)
    best_x = x.copy()
    
    # Adaptive beta: start low for exploration, increase for exploitation
    betas = [20, 50, 100, 200, 500]
    
    for beta in betas:
        def objective(x, beta=beta):
            c = x.reshape(N, 3)
            
            # Pairwise distances
            diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
            dists = np.sqrt((diff ** 2).sum(axis=2))
            np.fill_diagonal(dists, np.inf)
            pair_vals = dists / 2.0
            
            # Wall clearance
            abs_c = np.abs(c)
            max_coord = np.max(abs_c, axis=1)
            wall_vals = 0.5 - max_coord
            
            all_vals = np.concatenate([pair_vals.ravel(), wall_vals.ravel()])
            all_vals = all_vals[np.isfinite(all_vals)]
            
            if len(all_vals) == 0:
                return 0.0
            
            # Smooth min via log-sum-exp
            max_val = np.max(all_vals)
            shifted = -beta * (all_vals - max_val)
            shifted = np.clip(shifted, -700, 0)
            log_sum = np.log(np.sum(np.exp(shifted)))
            soft_min_val = max_val - log_sum / beta
            
            return -soft_min_val
        
        res = minimize(
            objective,
            x,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter, 'maxfev': max_iter * 10, 'ftol': 1e-12, 'gtol': 1e-10}
        )
        
        if res.x is not None:
            c_test = res.x.reshape(N, 3)
            r_test = _compute_radius(c_test)
            
            if r_test > best_r:
                best_r = r_test
                best_x = res.x.copy()
        
        x = res.x.copy()
    
    return best_x.reshape(N, 3), best_r

def construct():
    start_time = time.time()
    time_limit = 145.0  # Safety margin before 150s kill
    
    best_r = 0.0
    best_c = None
    
    # Try multiple initial configurations
    n_seeds = 6
    
    for seed in range(n_seeds):
        if time.time() - start_time > time_limit - 15:
            break
        
        rng = np.random.RandomState(seed)
        
        # Generate initial configuration
        if seed == 0:
            # 5x5x5 grid
            m = 5
            ticks = (np.arange(m) + 0.5) / m - 0.5
            grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
            centers = grid[:N].copy()
        elif seed == 1:
            # 4x5x5 grid with some rearrangement
            m1, m2, m3 = 4, 5, 5
            ticks1 = (np.arange(m1) + 0.5) / m1 - 0.5
            ticks2 = (np.arange(m2) + 0.5) / m2 - 0.5
            ticks3 = (np.arange(m3) + 0.5) / m3 - 0.5
            grid = np.array([(x, y, z) for z in ticks3 for y in ticks2 for x in ticks1])
            centers = grid[:N].copy()
        elif seed == 2:
            # Random uniform
            centers = rng.uniform(-0.4, 0.4, size=(N, 3))
        elif seed == 3:
            # 5x5x4 grid
            m1, m2, m3 = 5, 5, 4
            ticks1 = (np.arange(m1) + 0.5) / m1 - 0.5
            ticks2 = (np.arange(m2) + 0.5) / m2 - 0.5
            ticks3 = (np.arange(m3) + 0.5) / m3 - 0.5
            grid = np.array([(x, y, z) for z in ticks3 for y in ticks2 for x in ticks1])
            centers = grid[:N].copy()
        elif seed == 4:
            # Fibonacci sphere-like distribution projected to cube
            centers = rng.uniform(-0.45, 0.45, size=(N, 3))
        else:
            # Perturbed 5x5x5 grid
            m = 5
            ticks = (np.arange(m) + 0.5) / m - 0.5
            grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
            centers = grid[:N].copy()
            centers += rng.normal(0, 0.02, size=centers.shape)
            centers = np.clip(centers, -0.49, 0.49)
        
        # Phase 1: Force-directed relaxation to improve initial configuration
        if time.time() - start_time < time_limit - 20:
            # Try different target radii for force-directed relaxation
            for R_target in [0.08, 0.10, 0.11]:
                if time.time() - start_time > time_limit - 15:
                    break
                
                relaxed = _force_directed_relax(centers, R_target, n_steps=100, dt=0.0005, temp=0.005)
                r_test = _compute_radius(relaxed)
                
                if r_test > best_r:
                    best_r = r_test
                    best_c = relaxed.copy()
                
                # Use the relaxed configuration for further optimization
                centers = relaxed
        
        # Phase 2: Local refinement with adaptive beta
        if time.time() - start_time < time_limit - 10:
            refined_c, refined_r = _local_refine(centers, 0.11, max_iter=300)
            
            if refined_r > best_r:
                best_r = refined_r
                best_c = refined_c.copy()
        
        # Phase 3: Final high-precision polish
        if time.time() - start_time < time_limit - 5 and best_c is not None:
            # Use the best configuration found so far for final polish
            polished_c, polished_r = _local_refine(best_c, 0.11, max_iter=500)
            
            if polished_r > best_r:
                best_r = polished_r
                best_c = polished_c.copy()
    
    if best_c is None:
        # Fallback to simple grid
        m = 5
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        best_c = grid[:N]
    
    return best_c
