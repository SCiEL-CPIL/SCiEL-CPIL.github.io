"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _radius(centers):
    """Compute the maximum common radius for given centers."""
    n = len(centers)
    # Distance between all pairs
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    # Distance to walls: 1/2 - |c_i|
    wall_dists = 0.5 - np.abs(centers)
    min_wall = np.min(wall_dists)
    return min(min_pair / 2.0, min_wall)

def _radius_fast(centers):
    """Faster radius computation using vectorized operations."""
    n = len(centers)
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    wall_dists = 0.5 - np.abs(centers)
    min_wall = np.min(wall_dists)
    return min(min_pair / 2.0, min_wall)

def _soft_min_radius_neg(centers, beta):
    """Negative soft-min radius for optimization (to be minimized)."""
    n = len(centers)
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    
    # Wall distances
    wall_dists = 0.5 - np.abs(centers)
    wall_dists_flat = wall_dists.flatten()
    
    # Combine all constraints
    all_dists = np.concatenate([dists.flatten(), wall_dists_flat])
    all_dists = all_dists[all_dists > 0]  # Filter out zeros and negatives
    
    if len(all_dists) == 0:
        return 0.0
    
    # Soft minimum using log-sum-exp trick
    min_val = np.min(all_dists)
    # Clip for numerical stability
    clipped = np.clip(all_dists, min_val - 1e-10, None)
    soft_min = -np.log(np.mean(np.exp(-beta * clipped))) / beta + min_val - np.log(len(all_dists)) / beta
    
    # The actual radius is min(min_pair/2, min_wall)
    # So we need to be careful: pair distances contribute to r = d/2
    # Let's recompute properly
    
    # For pairs: constraint is d_ij >= 2r, so r <= d_ij/2
    # For walls: constraint is wall_dist >= r, so r <= wall_dist
    # So r = min(min_pair/2, min_wall)
    
    pair_dists = dists.flatten()
    pair_dists = pair_dists[pair_dists > 0]
    
    if len(pair_dists) == 0 or len(wall_dists_flat) == 0:
        return 0.0
    
    # Soft min of pair_dists/2 and wall_dists
    half_pairs = pair_dists / 2.0
    all_constraints = np.concatenate([half_pairs, wall_dists_flat])
    all_constraints = all_constraints[all_constraints > 0]
    
    if len(all_constraints) == 0:
        return 0.0
    
    min_val = np.min(all_constraints)
    clipped = np.clip(all_constraints, min_val - 1e-10, None)
    
    # Soft minimum
    log_sum = np.log(np.sum(np.exp(-beta * (clipped - min_val))))
    soft_min = min_val - log_sum / beta
    
    return -soft_min  # Negate for minimization

def _soft_min_radius(centers, beta):
    """Soft-min radius (positive value)."""
    return -_soft_min_radius_neg(centers, beta)

def _optimize_from_init(init, time_budget):
    """Optimize starting from an initial configuration."""
    start_time = time.time()
    
    best_centers = init.copy()
    best_radius = _radius_fast(init)
    
    # Progressive beta schedule
    betas = [10, 50, 100, 200, 500, 1000]
    
    current_centers = init.copy()
    
    for beta in betas:
        elapsed = time.time() - start_time
        if elapsed > time_budget * 0.8:
            break
        
        def objective(x):
            centers = x.reshape(-1, 2)
            return _soft_min_radius_neg(centers, beta)
        
        def gradient(x):
            centers = x.reshape(-1, 2)
            n = len(centers)
            
            # Compute soft-min gradient
            diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff**2, axis=-1))
            np.fill_diagonal(dists, np.inf)
            
            wall_dists = 0.5 - np.abs(centers)
            wall_dists_flat = wall_dists.flatten()
            
            half_pairs = dists.flatten() / 2.0
            half_pairs = half_pairs[half_pairs > 0]
            
            all_constraints = np.concatenate([half_pairs, wall_dists_flat])
            all_constraints = all_constraints[all_constraints > 0]
            
            if len(all_constraints) == 0:
                return np.zeros_like(x)
            
            min_val = np.min(all_constraints)
            clipped = np.clip(all_constraints, min_val - 1e-10, None)
            
            exp_vals = np.exp(-beta * (clipped - min_val))
            sum_exp = np.sum(exp_vals)
            weights = exp_vals / sum_exp
            
            # Gradient of soft-min with respect to constraints
            # d(soft_min)/d(constraint_i) = -weights_i
            # But we need gradient with respect to centers
            
            grad = np.zeros_like(centers)
            
            # Pair contributions
            pair_grad = np.zeros_like(centers)
            for i in range(n):
                for j in range(i+1, n):
                    d = centers[i] - centers[j]
                    dist = np.sqrt(np.sum(d**2))
                    if dist > 1e-10:
                        # Contribution to constraint dist/2
                        # d(dist/2)/d(centers[i]) = d/(2*dist)
                        # d(dist/2)/d(centers[j]) = -d/(2*dist)
                        weight = weights[0]  # This is wrong, need to track which weight
                        # Actually, let's use a simpler approach
                        pass
            
            # This gradient computation is complex. Let's use numerical gradient or a simpler approach.
            # For now, use L-BFGS-B which can compute gradients numerically if not provided.
            return None
        
        try:
            result = minimize(
                objective,
                current_centers.flatten(),
                method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * (2 * N),
                options={
                    'maxiter': 500,
                    'ftol': 1e-12,
                    'gtol': 1e-10,
                    'maxfun': 2000
                }
            )
            
            new_centers = result.x.reshape(-1, 2)
            new_radius = _radius_fast(new_centers)
            
            if new_radius > best_radius:
                best_radius = new_radius
                best_centers = new_centers.copy()
            
            current_centers = new_centers
            
        except Exception:
            pass
        
        elapsed = time.time() - start_time
        if elapsed > time_budget * 0.8:
            break
    
    # Final high-precision polish
    elapsed = time.time() - start_time
    if elapsed < time_budget * 0.9:
        try:
            def objective_final(x):
                centers = x.reshape(-1, 2)
                return _soft_min_radius_neg(centers, 1000)
            
            result = minimize(
                objective_final,
                best_centers.flatten(),
                method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * (2 * N),
                options={
                    'maxiter': 1000,
                    'ftol': 1e-14,
                    'gtol': 1e-12,
                    'maxfun': 5000
                }
            )
            
            new_centers = result.x.reshape(-1, 2)
            new_radius = _radius_fast(new_centers)
            
            if new_radius > best_radius:
                best_radius = new_radius
                best_centers = new_centers.copy()
                
        except Exception:
            pass
    
    return best_centers, best_radius

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill
    
    # Generate diverse initial configurations
    inits = []
    
    # 1. Grid
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    inits.append(grid[:N])
    
    # 2. Hexagonal lattice
    hex_init = []
    r_hex = 1.0 / (2 * np.sqrt(3) * 5)  # Approximate
    dx = r_hex * 2
    dy = r_hex * np.sqrt(3)
    rows = int(1.0 / dy) + 1
    for row in range(rows):
        offset = (row % 2) * dx / 2
        for col in range(int(1.0 / dx) + 1):
            x = -0.5 + dx / 2 + col * dx + offset
            y = -0.5 + dy / 2 + row * dy
            if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                hex_init.append([x, y])
    if len(hex_init) >= N:
        inits.append(np.array(hex_init[:N]))
    
    # 3. Random inits
    rng = np.random.RandomState(42)
    for _ in range(5):
        random_init = rng.uniform(-0.4, 0.4, (N, 2))
        inits.append(random_init)
    
    # 4. Concentric circles
    for num_rings in [3, 4, 5]:
        ring_init = [[0, 0]]
        remaining = N - 1
        for ring in range(1, num_rings):
            radius = ring * 0.1
            count = min(remaining, int(2 * np.pi * radius / 0.12))
            if count <= 0:
                continue
            angles = np.linspace(0, 2 * np.pi, count, endpoint=False)
            for angle in angles:
                ring_init.append([radius * np.cos(angle), radius * np.sin(angle)])
            remaining -= count
            if remaining <= 0:
                break
        while len(ring_init) < N:
            ring_init.append([rng.uniform(-0.4, 0.4), rng.uniform(-0.4, 0.4)])
        inits.append(np.array(ring_init[:N]))
    
    # Quick passes on all inits
    best_centers = inits[0].copy()
    best_radius = _radius_fast(inits[0])
    
    for init in inits:
        elapsed = time.time() - start_time
        if elapsed > deadline * 0.3:
            break
        
        try:
            centers, radius = _optimize_from_init(init, 5.0)
            if radius > best_radius:
                best_radius = radius
                best_centers = centers.copy()
        except Exception:
            pass
    
    # Deep optimization on best result
    elapsed = time.time() - start_time
    if elapsed < deadline * 0.5:
        try:
            centers, radius = _optimize_from_init(best_centers, deadline - elapsed - 5.0)
            if radius > best_radius:
                best_radius = radius
                best_centers = centers.copy()
        except Exception:
            pass
    
    # Additional restarts if time permits
    elapsed = time.time() - start_time
    if elapsed < deadline * 0.7:
        for _ in range(3):
            elapsed = time.time() - start_time
            if elapsed > deadline * 0.85:
                break
            
            # Perturb best solution
            perturbed = best_centers + np.random.randn(N, 2) * 0.05
            perturbed = np.clip(perturbed, -0.49, 0.49)
            
            try:
                centers, radius = _optimize_from_init(perturbed, 8.0)
                if radius > best_radius:
                    best_radius = radius
                    best_centers = centers.copy()
            except Exception:
                pass
    
    # Final validation
    final_radius = _radius_fast(best_centers)
    if not np.all(np.isfinite(best_centers)):
        return inits[0][:N]
    
    return best_centers
# EVOLVE-BLOCK-END
