"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time


def _radius(centers):
    """Compute the maximum common radius for given centers (vectorized)."""
    n = len(centers)
    # Pairwise distances (upper triangle)
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    d2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(d2, np.inf)
    min_pair_dist = np.sqrt(np.min(d2))
    # Distance to walls
    wall = 0.5 - np.max(np.abs(centers), axis=1)
    min_wall = np.min(wall)
    return min(min_pair_dist / 2.0, min_wall)


def _relax(centers, time_budget):
    """Force-directed relaxation to push spheres apart and away from walls."""
    start = time.time()
    centers = centers.copy()
    n = len(centers)
    best_r = _radius(centers)
    best = centers.copy()

    lr = 0.005
    decay = 0.9995
    while time.time() - start < time_budget:
        # Pairwise repulsion
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        d2 = np.sum(diff * diff, axis=2)
        np.fill_diagonal(d2, 1e-12)
        d = np.sqrt(d2)
        # Force magnitude ~ 1/d^2 (strong when close), softened
        mag = 1.0 / (d2 + 1e-3)
        forces = np.sum(mag[:, :, np.newaxis] * diff / d[:, :, np.newaxis], axis=1)

        # Wall repulsion
        margin = 0.02
        for k in range(3):
            over = centers[:, k] - (0.5 - margin)
            forces[:, k] -= np.where(over > 0, over * 5.0, 0.0)
            under = (-0.5 + margin) - centers[:, k]
            forces[:, k] += np.where(under > 0, under * 5.0, 0.0)

        centers += lr * forces
        centers = np.clip(centers, -0.4999, 0.4999)
        
        r = _radius(centers)
        if r > best_r:
            best_r = r
            best = centers.copy()
            
        lr *= decay

    return best, best_r

def _local_search(centers, time_budget):
    """Randomized local search: perturb random points, accept if better."""
    start = time.time()
    centers = centers.copy()
    best_r = _radius(centers)
    best = centers.copy()
    n = len(centers)
    step_size = 0.01
    
    while time.time() - start < time_budget:
        # Pick a random subset of points to move
        num_moves = 3
        idxs = np.random.choice(n, num_moves, replace=False)
        # Small random step
        delta = np.random.normal(0, step_size, (num_moves, 3))
        
        new_centers = centers.copy()
        new_centers[idxs] += delta
        new_centers = np.clip(new_centers, -0.4999, 0.4999)
        
        r = _radius(new_centers)
        if r > best_r:
            best_r = r
            best = new_centers.copy()
            centers = new_centers.copy()
            
    return best, best_r


def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin

    # Initial configuration: 4x4x4 grid (64 points), take 50
    m = 4
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    centers = grid[:N].copy()
    
    # The grid itself is a strong baseline.
    best_centers = centers.copy()
    best_r = _radius(best_centers)

    # Phase 1: Relaxation
    # We give it a small budget to see if it can improve the grid.
    if time.time() - start_time < deadline - 10:
        relaxed, r = _relax(centers, time_budget=5.0)
        if r > best_r:
            best_r = r
            best_centers = relaxed.copy()

    # Phase 2: Local Search
    # Try to polish the best configuration found so far.
    if time.time() - start_time < deadline - 5:
        budget = min(20.0, deadline - 5 - (time.time() - start_time))
        polished, r = _local_search(best_centers, budget)
        if r > best_r:
            best_r = r
            best_centers = polished.copy()

    # Final validity check
    if not np.all(np.isfinite(best_centers)) or np.any(np.abs(best_centers) > 0.5):
        best_centers = grid[:N].copy()

    return best_centers
# EVOLVE-BLOCK-END
