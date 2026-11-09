"""
Packing 50 equal circles in a unit circle.
Strategy:
1. Generate multiple diverse initial configurations (hexagonal lattice, ring-based, random with repulsion).
2. For each, run a physics-based optimization (Lennard-Jones potential) to maximize the minimum distance.
3. Return the best configuration found within the time budget.
"""
import numpy as np
import time
import sys

N = 50
TIME_LIMIT = 110.0  # Safety margin before 120s kill

def _score(points):
    """
    Compute the score for a given configuration.
    Score is the largest common radius r such that:
    1. All circles are inside the unit circle: r <= 1 - |c_i| for all i
    2. All circles are non-overlapping: r <= |c_i - c_j| / 2 for all i < j
    """
    if len(points) != N:
        return 0.0
    if not np.all(np.isfinite(points)):
        return 0.0
    
    # Distance to boundary
    dists_to_boundary = 1.0 - np.linalg.norm(points, axis=1)
    min_boundary = np.min(dists_to_boundary)
    
    if min_boundary <= 0:
        return 0.0
    
    # Pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    
    return min(min_boundary, min_pair)

def _lj_energy(points, r_target=None):
    """
    Compute Lennard-Jones-like energy for the configuration.
    We use a potential that encourages separation and keeps points inside.
    """
    n = len(points)
    if n < 2:
        return 0.0
    
    # Pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, 1.0)  # Avoid division by zero
    
    # Lennard-Jones potential: 4 * ((1/d)^12 - (1/d)^6)
    # But we scale it to be more stable
    # We want to maximize the minimum distance, so we minimize a potential that penalizes close pairs
    # Using a softer potential: sum of 1/d^2 for pairs closer than some threshold
    # Actually, let's use a standard LJ with sigma = 1 for now, but scaled
    
    # To make it more effective for packing, we use:
    # E = sum_{i<j} (1/d_ij^12 - 1/d_ij^6) + lambda * sum_i max(0, |c_i| - (1-r))^2
    # But we don't know r. Instead, we just minimize the LJ energy which tends to spread points.
    
    # Simpler approach: minimize sum of 1/d_ij^4 for all pairs (repulsive only)
    # This is more stable and encourages uniform distribution
    inv_d2 = 1.0 / (dists ** 2)
    np.fill_diagonal(inv_d2, 0.0)
    
    # Boundary penalty: penalize points too close to the edge
    r_norms = np.linalg.norm(points, axis=1)
    boundary_penalty = np.sum(np.maximum(0, r_norms - 0.99) ** 4)
    
    energy = np.sum(inv_d2) + 100.0 * boundary_penalty
    return energy

def _optimize(points, max_iters=5000, lr_init=0.01, time_budget=None):
    """
    Optimize the configuration using gradient descent on a repulsive potential.
    """
    start_time = time.time()
    n = len(points)
    points = points.copy()
    lr = lr_init
    best_energy = _lj_energy(points)
    best_points = points.copy()
    patience = 0
    
    for it in range(max_iters):
        if time_budget is not None and time.time() - start_time > time_budget:
            break
        
        # Compute forces (negative gradient of energy)
        # Force on i from j: -dE/dd_i = -d/dd_i (1/d_ij^2) = 2 * (r_i - r_j) / d_ij^4
        # Wait, E = sum 1/d^2, dE/dd = -2/d^3 * d(d)/d(r_i)
        # d(r_ij)/d(r_i) = (r_i - r_j)/d_ij
        # So dE/dr_i = sum_j -2/d_ij^3 * (r_i - r_j)/d_ij = -2 * sum_j (r_i - r_j)/d_ij^4
        # Force = -dE/dr_i = 2 * sum_j (r_i - r_j)/d_ij^4
        
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists_sq = np.sum(diff ** 2, axis=2)
        np.fill_diagonal(dists_sq, 1.0)
        
        # 1/d^4 = 1/dists_sq^2
        inv_d4 = 1.0 / (dists_sq ** 2)
        np.fill_diagonal(inv_d4, 0.0)
        
        forces = 2.0 * np.sum(diff * inv_d4[:, :, np.newaxis], axis=1)
        
        # Boundary force: push points inward if they are too close to the edge
        r_norms = np.linalg.norm(points, axis=1)
        for i in range(n):
            if r_norms[i] > 0.95:
                # Push inward
                force_boundary = -10.0 * (r_norms[i] - 0.95) * points[i] / r_norms[i]
                forces[i] += force_boundary
        
        # Update
        points += lr * forces
        points = np.clip(points, -1.5, 1.5)  # Keep in reasonable range
        
        # Compute energy
        energy = _lj_energy(points)
        
        if energy < best_energy:
            best_energy = energy
            best_points = points.copy()
            patience = 0
        else:
            patience += 1
            # Reduce learning rate
            lr *= 0.95
        
        if patience > 200:
            break
    
    return best_points

def _hexagonal_init(n_points):
    """Generate a hexagonal lattice configuration."""
    points = []
    spacing = 0.2
    for k in range(10):
        for i in range(-k, k + 1):
            for j in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3) / 2
                if np.hypot(x, y) < 1.0:
                    points.append([x, y])
    
    points = np.array(points)
    if len(points) > n_points:
        # Sort by distance from center and take closest
        dists = np.linalg.norm(points, axis=1)
        idx = np.argsort(dists)[:n_points]
        points = points[idx]
    
    while len(points) < n_points:
        # Add random points
        theta = np.random.uniform(0, 2 * np.pi)
        r = np.random.uniform(0, 0.8)
        points = np.vstack([points, [r * np.cos(theta), r * np.sin(theta)]])
    
    return points[:n_points]

def _ring_init(n_points):
    """Generate a ring-based configuration."""
    points = [[0.0, 0.0]]
    remaining = n_points - 1
    ring_idx = 1
    while remaining > 0:
        # Number of points on this ring
        n_ring = min(6 * ring_idx, remaining)
        radius = 0.15 * ring_idx
        for i in range(n_ring):
            theta = 2 * np.pi * i / n_ring
            x = radius * np.cos(theta)
            y = radius * np.sin(theta)
            points.append([x, y])
        remaining -= n_ring
        ring_idx += 1
    
    return np.array(points[:n_points])

def _random_init(n_points, seed=None):
    """Generate a random configuration with repulsion."""
    if seed is not None:
        np.random.seed(seed)
    
    points = np.zeros((n_points, 2))
    # Place center
    points[0] = [0.0, 0.0]
    
    for i in range(1, n_points):
        for attempt in range(1000):
            theta = np.random.uniform(0, 2 * np.pi)
            r = np.random.uniform(0.05, 0.95)
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            
            # Check distance to existing points
            dists = np.linalg.norm(points[:i] - [x, y], axis=1)
            if np.min(dists) > 0.15:  # Minimum separation
                points[i] = [x, y]
                break
        
        if np.min(np.linalg.norm(points[:i] - points[i], axis=1)) < 0.15:
            # Just place it randomly
            theta = np.random.uniform(0, 2 * np.pi)
            r = np.random.uniform(0.05, 0.95)
            points[i] = [r * np.cos(theta), r * np.sin(theta)]
    
    return points

def construct():
    start_time = time.time()
    
    # Generate multiple initial configurations
    inits = []
    
    # Hexagonal
    inits.append(_hexagonal_init(N))
    
    # Ring-based
    inits.append(_ring_init(N))
    
    # Random with different seeds
    for seed in [42, 123, 456, 789]:
        inits.append(_random_init(N, seed=seed))
    
    best_score = 0.0
    best_points = inits[0]
    
    for init_idx, init in enumerate(inits):
        elapsed = time.time() - start_time
        if elapsed > TIME_LIMIT - 10:
            break
        
        # Allocate time for this optimization
        time_per_opt = (TIME_LIMIT - 10 - elapsed) / (len(inits) - init_idx)
        time_per_opt = max(time_per_opt, 5.0)  # Minimum 5 seconds per optimization
        
        optimized = _optimize(init, max_iters=3000, lr_init=0.005, time_budget=time_per_opt)
        
        score = _score(optimized)
        if score > best_score:
            best_score = score
            best_points = optimized
        
        elapsed = time.time() - start_time
        if elapsed > TIME_LIMIT - 5:
            break
    
    # Final check
    final_score = _score(best_points)
    if final_score <= 0:
        # Fallback to hexagonal
        best_points = _hexagonal_init(N)
    
    return best_points
