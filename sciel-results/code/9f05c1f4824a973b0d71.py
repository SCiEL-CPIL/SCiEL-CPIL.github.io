"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_radius(centres):
    """Compute the maximum common radius for given centres."""
    # Pairwise distances using vectorized computation
    diff = centres[:, np.newaxis, :] - centres[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=-1))
    np.fill_diagonal(dist, np.inf)
    min_pair_dist = np.min(dist)
    
    # Distance to walls
    wall_dist = 0.5 - np.abs(centres)
    min_wall_dist = np.min(wall_dist)
    
    return min(min_pair_dist / 2.0, min_wall_dist)


def _soft_min_radius(centres, beta=100.0):
    """
    Smooth surrogate for the minimum radius using log-sum-exp.
    Returns a smooth approximation of min(min_pair_dist/2, min_wall_dist).
    """
    # Pairwise distances
    diff = centres[:, np.newaxis, :] - centres[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(dist_sq, np.inf)
    
    # Wall distances squared (we work with distances, not squared)
    wall_dist = 0.5 - np.abs(centres)
    
    # We want soft_min of (dist/2 for all pairs) and (wall_dist for all coords)
    # Let's compute soft min of all constraints
    # dist/2 for pairs: there are N*(N-1)/2 unique pairs, but we have full matrix
    # Let's just use the full dist matrix (excluding diagonal)
    
    # For the soft min, we need to combine:
    # - all pairwise distances / 2
    # - all wall distances
    
    # Flatten pairwise distances (excluding diagonal)
    mask = ~np.eye(len(centres), dtype=bool)
    pair_dists = dist_sq[mask] / 4.0  # squared distance / 4 = (dist/2)^2
    # Actually, we need dist/2, not (dist/2)^2. Let's use sqrt.
    # But sqrt is expensive. Let's just use the actual distances.
    
    # Recompute with actual distances for clarity
    dist = np.sqrt(dist_sq)
    np.fill_diagonal(dist, np.inf)
    pair_dists = dist[mask] / 2.0
    
    wall_dists = wall_dist.flatten()
    
    # Combine all constraints
    all_dists = np.concatenate([pair_dists, wall_dists])
    
    # Soft min: -beta * log(sum(exp(-beta * x)))
    # For numerical stability, subtract max
    x_max = np.max(all_dists)
    log_sum = np.log(np.sum(np.exp(-beta * (all_dists - x_max))))
    soft_min_val = x_max - np.log(np.sum(np.exp(-beta * (all_dists - x_max)))) / beta
    
    return soft_min_val


def _vectorized_force_directed(centres, time_limit=100.0):
    """
    Fully vectorized force-directed optimization with smooth surrogate objective.
    """
    start_time = time.time()
    best_centres = centres.copy()
    best_radius = _compute_radius(centres)
    
    n = len(centres)
    c = centres.copy()
    
    step = 0.003
    min_step = 1e-8
    patience = 0
    max_patience = 200
    beta = 50.0  # Smoothness parameter for soft-min
    
    while time.time() - start_time < time_limit:
        # Compute all pairwise distances (vectorized)
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dist_sq = np.sum(diff * diff, axis=-1)
        dist = np.sqrt(dist_sq)
        np.fill_diagonal(dist, np.inf)
        
        # Compute forces using smooth surrogate gradient
        # The soft-min gradient points in the direction of increasing the minimum constraint
        # For each pair (i,j) with distance d_ij, the contribution to force on i is:
        # proportional to exp(-beta * d_ij/2) * (c_i - c_j) / d_ij
        # Similarly for walls
        
        # Pairwise forces
        threshold = 0.4
        mask = (dist < threshold) & (dist > 1e-10)
        
        # Weight by exp(-beta * dist/2) for smooth min approximation
        weights = np.zeros_like(dist)
        weights[mask] = np.exp(-beta * dist[mask] / 2.0)
        
        # Force: sum_j weights[i,j] * (c_i - c_j) / d_ij
        # = sum_j weights[i,j] * diff[i,j] / dist[i,j]
        inv_d = np.zeros_like(dist)
        inv_d[mask] = 1.0 / dist[mask]
        
        forces = np.sum(diff * (weights * inv_d)[:, :, np.newaxis], axis=1)
        
        # Wall forces using smooth approximation
        for k in range(3):
            wall_d = 0.5 - np.abs(c[:, k])
            # Weight by exp(-beta * wall_d)
            w = np.exp(-beta * wall_d)
            # Force: if c > 0, push left; if c < 0, push right
            # Gradient of wall distance: -sign(c) for c>0, +sign(c) for c<0
            # Actually, wall distance = 0.5 - |c|, so d(wall)/dc = -sign(c)
            # Force should increase wall distance, so force = -d(wall)/dc * w = sign(c) * w
            # But we want to push AWAY from wall, so:
            # If c > 0.45, push left (negative)
            # If c < -0.45, push right (positive)
            # This is: -sign(c) * w * (c - 0.5*sign(c)) when close to wall
            # Simpler: just use the sign
            forces[:, k] -= np.sign(c[:, k]) * w * (0.5 - np.abs(c[:, k]))
        
        # Weak centering force
        forces -= 0.005 * c
        
        # Normalize forces for stable stepping
        force_mags = np.linalg.norm(forces, axis=1, keepdims=True)
        force_mags = np.maximum(force_mags, 1e-10)
        forces = forces / force_mags
        
        # Apply forces
        c += step * forces
        
        # Clip to valid range
        c = np.clip(c, -0.5 + 1e-9, 0.5 - 1e-9)
        
        # Check improvement using exact radius
        current_radius = _compute_radius(c)
        if current_radius > best_radius:
            best_radius = current_radius
            best_centres = c.copy()
            patience = 0
            step = min(step * 1.02, 0.01)
        else:
            patience += 1
            step = max(step * 0.99, min_step)
        
        # Random perturbation if stuck
        if patience > max_patience:
            rng = np.random.RandomState(int(time.time() * 1000) % 2**31)
            c += rng.normal(0, 0.008, size=c.shape)
            c = np.clip(c, -0.5 + 1e-9, 0.5 - 1e-9)
            patience = 0
            step = 0.003
    
    return best_centres, best_radius


def _make_fcc_packing():
    """
    Create an FCC-inspired packing for 50 spheres.
    """
    # FCC basis
    basis = np.array([
        [0, 0, 0],
        [0.5, 0.5, 0],
        [0.5, 0, 0.5],
        [0, 0.5, 0.5]
    ])
    
    # Generate FCC lattice points in a 3x3x4 supercell
    nx, ny, nz = 3, 3, 4
    points = []
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                for b in basis:
                    p = np.array([ix, iy, iz]) + b
                    p = (p - 1.25) / 2.5
                    points.append(p)
    
    points = np.array(points)
    
    # Select 50 with maximum minimum distance using greedy algorithm
    selected = [0]
    remaining = list(range(1, len(points)))
    
    while len(selected) < N and remaining:
        best_idx = -1
        best_min_dist = -1
        for idx in remaining:
            min_d = np.min(np.linalg.norm(points[selected] - points[idx], axis=1))
            if min_d > best_min_dist:
                best_min_dist = min_d
                best_idx = idx
        
        if best_idx == -1:
            break
        selected.append(best_idx)
        remaining.remove(best_idx)
    
    centres = points[selected]
    
    # Add small random perturbation to break symmetry
    rng = np.random.RandomState(42)
    centres += rng.normal(0, 0.003, size=centres.shape)
    centres = np.clip(centres, -0.49, 0.49)
    
    return centres


def _make_random_packing(seed):
    """Create a random initial packing."""
    rng = np.random.RandomState(seed)
    centres = rng.uniform(-0.45, 0.45, size=(N, 3))
    return centres


def _make_grid_packing():
    """Create a grid-based initial packing."""
    m = 4
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    
    selected = [0]
    remaining = list(range(1, len(grid)))
    
    while len(selected) < N and remaining:
        best_idx = -1
        best_min_dist = -1
        for idx in remaining:
            min_d = np.min(np.linalg.norm(grid[selected] - grid[idx], axis=1))
            if min_d > best_min_dist:
                best_min_dist = min_d
                best_idx = idx
        
        if best_idx == -1:
            break
        selected.append(best_idx)
        remaining.remove(best_idx)
    
    centres = grid[selected]
    
    rng = np.random.RandomState(123)
    centres += rng.normal(0, 0.002, size=centres.shape)
    centres = np.clip(centres, -0.49, 0.49)
    
    return centres


def _polish(centres, time_limit=10.0):
    """
    High-precision polish pass with very small steps and tighter convergence.
    """
    start_time = time.time()
    best_centres = centres.copy()
    best_radius = _compute_radius(centres)
    
    n = len(centres)
    c = centres.copy()
    
    step = 0.0002
    min_step = 1e-12
    patience = 0
    max_patience = 1000
    
    while time.time() - start_time < time_limit:
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dist_sq = np.sum(diff * diff, axis=-1)
        dist = np.sqrt(dist_sq)
        np.fill_diagonal(dist, np.inf)
        
        threshold = 0.15
        mask = (dist < threshold) & (dist > 1e-10)
        
        # Use inverse square for stronger repulsion at close range
        inv_d2 = np.zeros_like(dist)
        inv_d2[mask] = 1.0 / (dist[mask] ** 2)
        
        forces = np.sum(diff * inv_d2[:, :, np.newaxis] / dist[:, :, np.newaxis], axis=1)
        # Simplify: diff / d^3
        forces = np.sum(diff * inv_d2[:, :, np.newaxis], axis=1) / 1.0  # This is wrong, let me fix
        
        # Correct: force = diff / d^3
        inv_d3 = np.zeros_like(dist)
        inv_d3[mask] = 1.0 / (dist[mask] ** 3)
        forces = np.sum(diff * inv_d3[:, :, np.newaxis], axis=1)
        
        for k in range(3):
            above = c[:, k] > 0.49
            below = c[:, k] < -0.49
            forces[above, k] -= (c[above, k] - 0.49) * 200.0
            forces[below, k] += (-0.49 - c[below, k]) * 200.0
        
        forces -= 0.0005 * c
        
        force_mags = np.linalg.norm(forces, axis=1, keepdims=True)
        force_mags = np.maximum(force_mags, 1e-10)
        forces = forces / force_mags
        
        c += step * forces
        c = np.clip(c, -0.5 + 1e-10, 0.5 - 1e-10)
        
        current_radius = _compute_radius(c)
        if current_radius > best_radius:
            best_radius = current_radius
            best_centres = c.copy()
            patience = 0
        else:
            patience += 1
            step = max(step * 0.97, min_step)
        
        if patience > max_patience:
            break
    
    return best_centres, best_radius


def construct():
    """Construct centres of N equal spheres in unit cube."""
    start_time = time.time()
    deadline = start_time + 145.0  # Leave 5s margin for safety
    
    best_centres = None
    best_radius = 0.0
    
    # Strategy 1: FCC-inspired packing
    if time.time() < deadline - 40:
        try:
            init = _make_fcc_packing()
            centres, radius = _vectorized_force_directed(init, time_limit=min(50, deadline - time.time()))
            if radius > best_radius:
                best_radius = radius
                best_centres = centres
        except Exception:
            pass
    
    # Strategy 2: Grid-based packing
    if time.time() < deadline - 30 and best_centres is not None:
        try:
            init = _make_grid_packing()
            centres, radius = _vectorized_force_directed(init, time_limit=min(35, deadline - time.time()))
            if radius > best_radius:
                best_radius = radius
                best_centres = centres
        except Exception:
            pass
    
    # Strategy 3: Multiple random restarts with short optimizations
    if time.time() < deadline - 20:
        for seed in [1, 7, 13, 42, 99, 123, 456, 789, 1000, 2024, 555, 666, 777, 888, 999]:
            if time.time() > deadline - 10:
                break
            try:
                init = _make_random_packing(seed)
                time_left = deadline - time.time() - 5
                if time_left < 5:
                    break
                centres, radius = _vectorized_force_directed(init, time_limit=min(10, time_left))
                if radius > best_radius:
                    best_radius = radius
                    best_centres = centres
            except Exception:
                continue
    
    # Final polish pass - use remaining time
    if best_centres is not None and time.time() < deadline - 1:
        try:
            polish_time = min(15, deadline - time.time() - 0.5)
            polished, pr = _polish(best_centres, time_limit=polish_time)
            if pr > best_radius:
                best_radius = pr
                best_centres = polished
        except Exception:
            pass
    
    # If no valid result, fallback to simple grid
    if best_centres is None:
        m = 4
        ticks = (np.arange(m) + 0.5) / m - 0.5
        best_centres = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])[:N]
    
    # Final validation
    if not np.all(np.isfinite(best_centres)) or np.any(np.abs(best_centres) > 0.5):
        m = 4
        ticks = (np.arange(m) + 0.5) / m - 0.5
        best_centres = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])[:N]
    
    return best_centres
# EVOLVE-BLOCK-END
