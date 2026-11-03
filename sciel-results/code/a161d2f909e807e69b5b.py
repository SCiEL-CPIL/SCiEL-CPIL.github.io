"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _compute_radius(centers):
    """Compute the maximum common radius for given centers."""
    centers = np.asarray(centers)
    if len(centers) < 2:
        return float(np.min(0.5 - np.abs(centers)))
    
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    sq_dists = np.sum(diff**2, axis=2)
    np.fill_diagonal(sq_dists, np.inf)
    min_pair_dist = np.sqrt(np.min(sq_dists))
    
    min_wall_dist = np.min(0.5 - np.abs(centers))
    
    return float(min(min_pair_dist / 2.0, min_wall_dist))

def _energy_and_gradient(x, r_target, softness=0.01):
    """
    Compute smooth energy that penalizes overlaps and wall violations.
    Energy = sum over pairs of max(0, r_target - d_ij/2)^2 / softness^2
           + sum over walls of max(0, r_target - wall_dist)^2 / softness^2
    
    Returns (energy, gradient) where x is flat array of length N*3.
    """
    centers = x.reshape(N, 3)
    n = len(centers)
    grad = np.zeros_like(centers)
    energy = 0.0
    
    # Pairwise repulsion: penalize when d_ij < 2*r_target
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    sq_dists = np.sum(diff**2, axis=2)
    np.fill_diagonal(sq_dists, np.inf)
    
    # Only consider pairs that are close (within 2*r_target + margin)
    margin = 4 * softness
    close_mask = sq_dists < (2 * r_target + margin)**2
    
    if np.any(close_mask):
        # For close pairs, compute smooth penalty
        # penalty_ij = max(0, 2*r_target - d_ij)^2 / (4 * softness^2)
        # Use soft approximation: (2*r_target - d_ij)^2 / (4*softness^2) when close
        # But we need smoothness, so use: for d < 2*r_target, penalty = (2*r_target - d)^2
        # Gradient w.r.t. centers_i: -2*(2*r_target - d) * (centers_i - centers_j) / d
        
        d = np.sqrt(np.maximum(sq_dists, 1e-20))
        violation = 2 * r_target - d  # positive when overlapping
        
        # Smooth: only apply where violation > -margin
        active = close_mask & (violation > -margin)
        
        if np.any(active):
            # Scale factor for smoothness
            # Use: penalty = violation^2 / (4 * softness^2) * sigmoid-like smooth
            # Simpler: just use violation^2 for active pairs, scaled
            scale = 1.0 / (4 * softness**2)
            
            # Energy contribution
            active_violation = np.where(active, violation, 0.0)
            energy += np.sum(active_violation**2) * scale
            
            # Gradient: d/d(centers_i) [sum_j (2r - d_ij)^2 / (4*softness^2)]
            # = sum_j (2r - d_ij) * (-1) * (centers_i - centers_j) / d_ij / (2*softness^2)
            # = -1/(2*softness^2) * sum_j violation_ij * (centers_i - centers_j) / d_ij
            
            # Build gradient matrix
            unit_diff = np.zeros_like(diff)
            valid_d = d > 1e-10
            unit_diff[valid_d] = diff[valid_d] / d[valid_d, np.newaxis]
            
            # grad_i = -1/(2*softness^2) * sum_j violation_ij * unit_diff_ij
            # But only for active pairs
            active_unit = np.where(active[:, :, np.newaxis], unit_diff, 0.0)
            active_viol = np.where(active, violation, 0.0)
            
            grad_pair = -1.0 / (2 * softness**2) * np.sum(active_viol[:, :, np.newaxis] * active_unit, axis=1)
            grad += grad_pair
    
    # Wall constraints: penalize when wall_dist < r_target
    wall_dists = 0.5 - np.abs(centers)  # shape (N, 3)
    wall_violation = r_target - wall_dists  # positive when sphere hits wall
    
    wall_active = wall_violation > -margin
    if np.any(wall_active):
        active_wall_viol = np.where(wall_active, wall_violation, 0.0)
        energy += np.sum(active_wall_viol**2) * scale
        
        # Gradient: d/d(centers_ik) [(r - (0.5 - |c_ik|))^2 / (4*softness^2)]
        # = 2*(r - wall_ik) * sign(c_ik) / (4*softness^2)
        # = (r - wall_ik) * sign(c_ik) / (2*softness^2)
        signs = np.sign(centers)
        signs[centers == 0] = 1.0
        grad_wall = active_wall_viol * signs / (2 * softness**2)
        grad += grad_wall
    
    return float(energy), grad.flatten()

def _minimize_energy(init_centers, r_target, softness=0.01, max_iter=2000, time_limit=10.0):
    """Minimize energy for given target radius using L-BFGS-B."""
    start_time = time.time()
    x0 = init_centers.flatten()
    
    def obj(x):
        if time.time() - start_time > time_limit:
            return 1e10, np.zeros_like(x)
        val, grad = _energy_and_gradient(x, r_target, softness)
        return val, grad
    
    try:
        result = minimize(
            obj, x0,
            method='L-BFGS-B',
            bounds=[(-0.5, 0.5)] * (N * 3),
            options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
        )
        return result.x.reshape(N, 3)
    except Exception:
        return init_centers.copy()

def _iterative_radius_increase(centers, r_start, r_max, n_steps=50, time_limit=60.0):
    """
    Iteratively increase target radius and minimize energy.
    Returns (best_centers, best_radius).
    """
    start_time = time.time()
    best_centers = centers.copy()
    best_radius = _compute_radius(centers)
    
    # Create a schedule of target radii
    r_values = np.linspace(r_start, r_max, n_steps)
    
    for r_target in r_values:
        if time.time() - start_time > time_limit:
            break
        
        # Try several softness values, decreasing over iterations
        progress = (r_target - r_start) / (r_max - r_start + 1e-10)
        softness = max(0.001, 0.05 * (1 - progress) + 0.001)
        
        time_remaining = time_limit - (time.time() - start_time)
        if time_remaining < 2.0:
            break
        
        try:
            optimized = _minimize_energy(
                best_centers, r_target, 
                softness=softness,
                max_iter=1500,
                time_limit=min(8.0, time_remaining - 1.0)
            )
            r_actual = _compute_radius(optimized)
            if r_actual > best_radius:
                best_radius = r_actual
                best_centers = optimized.copy()
        except Exception:
            continue
    
    return best_centers, best_radius

def _polish_critical_pairs(centers, time_limit=5.0):
    """Final high-precision polish focusing only on critical (closest) pairs."""
    start_time = time.time()
    centers = centers.copy()
    
    for iteration in range(500):
        if time.time() - start_time > time_limit:
            break
        
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        sq_dists = np.sum(diff**2, axis=2)
        np.fill_diagonal(sq_dists, np.inf)
        
        min_sq_dist = np.min(sq_dists)
        min_pair_dist = np.sqrt(min_sq_dist)
        min_pair_idx = np.unravel_index(np.argmin(sq_dists), sq_dists.shape)
        
        min_wall_dist = np.min(0.5 - np.abs(centers))
        min_wall_idx = np.unravel_index(np.argmin(0.5 - np.abs(centers)), (N, 3))
        
        r = min(min_pair_dist / 2.0, min_wall_dist)
        
        if min_pair_dist / 2.0 < min_wall_dist:
            i, j = min_pair_idx
            if i > j:
                i, j = j, i
            d = centers[i] - centers[j]
            dist = np.linalg.norm(d)
            if dist < 1e-10:
                d = np.array([1e-6, 0, 0])
                dist = 1e-6
            push = (min_pair_dist / 2.0 - r) * d / dist * 0.5
            if push > 0:
                centers[i] += push
                centers[j] -= push
        else:
            i, k = min_wall_idx
            wall = 0.5 - np.abs(centers[i, k])
            if wall < r:
                sign = np.sign(centers[i, k])
                if sign == 0:
                    sign = 1.0
                centers[i, k] += (r - wall) * sign
        
        centers = np.clip(centers, -0.5, 0.5)
    
    return centers

def _grid_init(n):
    """Generate grid initial configuration."""
    m = int(np.ceil(n ** (1.0/3.0)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return grid[:n]

def _bcc_init(n):
    """Generate body-centered cubic lattice initial configuration."""
    m = int(np.ceil((n / 2) ** (1.0/3.0)))
    spacing = 1.0 / m
    centers = []
    for ix in range(m):
        for iy in range(m):
            for iz in range(m):
                x = (ix + 0.5) * spacing - 0.5
                y = (iy + 0.5) * spacing - 0.5
                z = (iz + 0.5) * spacing - 0.5
                centers.append([x, y, z])
                if ix < m-1 and iy < m-1 and iz < m-1:
                    bx = (ix + 1.0) * spacing - 0.5
                    by = (iy + 1.0) * spacing - 0.5
                    bz = (iz + 1.0) * spacing - 0.5
                    centers.append([bx, by, bz])
    
    centers = np.array(centers)
    if len(centers) > n:
        centers = centers[:n]
    elif len(centers) < n:
        grid = _grid_init(n)
        centers = np.vstack([centers, grid[len(centers):n]])
    
    return centers

def _random_init(n, seed=None):
    """Generate random initial configuration inside the cube."""
    rng = np.random.RandomState(seed)
    return rng.uniform(-0.45, 0.45, size=(n, 3))

def _jittered_grid_init(n, seed=None):
    """Generate jittered grid initial configuration."""
    grid = _grid_init(n)
    rng = np.random.RandomState(seed)
    jitter = rng.uniform(-0.02, 0.02, size=(n, 3))
    return np.clip(grid + jitter, -0.49, 0.49)

def construct():
    """Construct the optimal packing of N spheres in the unit cube."""
    start_time = time.time()
    deadline = start_time + 140.0  # 10 second safety margin
    
    best_centers = None
    best_radius = 0.0
    
    # Generate initializations
    inits = []
    grid = _grid_init(N)
    inits.append(('grid', grid))
    
    bcc = _bcc_init(N)
    inits.append(('bcc', bcc))
    
    for seed in range(4):
        inits.append((f'rand{seed}', _random_init(N, seed=seed)))
    
    for seed in range(4):
        inits.append((f'jitter{seed}', _jittered_grid_init(N, seed=seed)))
    
    # Evaluate all inits
    for name, init in inits:
        r = _compute_radius(init)
        if r > best_radius:
            best_radius = r
            best_centers = init.copy()
    
    # Stage 1: Iterative radius increase from multiple initializations
    # Use the best few initializations
    active_inits = [best_centers]
    for name, init in inits[:4]:
        if init is not best_centers:
            active_inits.append(init)
    
    for init in active_inits:
        if time.time() > deadline - 50.0:
            break
        
        try:
            r_start = _compute_radius(init)
            r_max = min(0.115, r_start * 1.5)  # Don't jump too far
            time_remaining = deadline - time.time() - 20.0
            if time_remaining < 10.0:
                break
            
            optimized, r_opt = _iterative_radius_increase(
                init, r_start, r_max,
                n_steps=30,
                time_limit=min(40.0, time_remaining)
            )
            if r_opt > best_radius:
                best_radius = r_opt
                best_centers = optimized.copy()
        except Exception:
            continue
    
    # Stage 2: Fine polishing with small softness
    if time.time() < deadline - 20.0:
        try:
            r_start = best_radius
            r_max = min(0.115, r_start * 1.02)
            time_remaining = deadline - time.time() - 10.0
            if time_remaining > 5.0:
                optimized, r_opt = _iterative_radius_increase(
                    best_centers, r_start, r_max,
                    n_steps=20,
                    time_limit=min(15.0, time_remaining)
                )
                if r_opt > best_radius:
                    best_radius = r_opt
                    best_centers = optimized.copy()
        except Exception:
            pass
    
    # Stage 3: Final critical-pair polish
    if time.time() < deadline - 5.0:
        try:
            polished = _polish_critical_pairs(best_centers, time_limit=4.0)
            r_polished = _compute_radius(polished)
            if r_polished > best_radius:
                best_radius = r_polished
                best_centers = polished
        except Exception:
            pass
    
    # Final validation and clipping
    if best_centers is None:
        best_centers = _grid_init(N)
    
    best_centers = np.clip(best_centers, -0.4999999, 0.4999999)
    
    return best_centers
# EVOLVE-BLOCK-END
