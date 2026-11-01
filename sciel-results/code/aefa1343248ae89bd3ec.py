"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import cdist

def _score(centers):
    """Compute the radius."""
    centers = np.asarray(centers, dtype=np.float64)
    if centers.shape != (N, 3) or not np.all(np.isfinite(centers)):
        return -1.0
    wall = 0.5 - np.abs(centers)
    wall_min = float(np.min(wall))
    if wall_min < -1e-12:
        return -1.0
    d = cdist(centers, centers)
    np.fill_diagonal(d, np.inf)
    pair_min = float(np.min(d)) / 2.0
    return min(wall_min, pair_min)

def _repulsion_step(centers, dt):
    """Vectorized repulsion + wall push."""
    n = len(centers)
    # Pairwise differences: (n, n, 3)
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    # Avoid division by zero
    dist_safe = np.where(dist < 1e-10, 1e-10, dist)
    # Force magnitude: 1/(dist^2 + eps), zero on diagonal
    f_mag = 1.0 / (dist_safe ** 2 + 0.005)
    np.fill_diagonal(f_mag, 0.0)
    # Force direction
    f_dir = diff / dist_safe[:, :, np.newaxis]
    forces = np.sum(f_mag[:, :, np.newaxis] * f_dir, axis=1)
    
    # Wall repulsion
    wall_force = np.zeros_like(centers)
    for k in range(3):
        d_left = centers[:, k] - (-0.5)
        d_right = 0.5 - centers[:, k]
        # Push away from walls when close
        wall_force[:, k] += np.where(d_left < 0.15, 8.0 * (0.15 - d_left), 0.0)
        wall_force[:, k] -= np.where(d_right < 0.15, 8.0 * (0.15 - d_right), 0.0)
    
    centers = centers + dt * (forces + wall_force)
    centers = np.clip(centers, -0.5, 0.5)
    return centers

def _local_refine(centers, max_iter=3000, time_limit=None):
    """Hill-climb on the min-distance objective."""
    centers = centers.copy()
    current_score = _score(centers)
    step = 0.003
    for it in range(max_iter):
        if time_limit is not None and time.time() > time_limit:
            break
        # Find limiting constraint
        d = cdist(centers, centers)
        np.fill_diagonal(d, np.inf)
        pair_min = float(np.min(d)) / 2.0
        wall = 0.5 - np.abs(centers)
        wall_min = float(np.min(wall))
        
        improved = False
        if pair_min <= wall_min:
            idx = np.unravel_index(np.argmin(d), d.shape)
            i, j = int(idx[0]), int(idx[1])
            diff = centers[i] - centers[j]
            dist = np.linalg.norm(diff)
            if dist < 1e-12:
                continue
            direction = diff / dist
            new_i = np.clip(centers[i] + step * direction, -0.5, 0.5)
            old = centers[i].copy()
            centers[i] = new_i
            new_score = _score(centers)
            if new_score > current_score:
                current_score = new_score
                improved = True
            else:
                centers[i] = old
        else:
            idx = np.unravel_index(np.argmin(wall), wall.shape)
            i, k = int(idx[0]), int(idx[1])
            if centers[i, k] > 0:
                direction = np.zeros(3); direction[k] = -1.0
            else:
                direction = np.zeros(3); direction[k] = 1.0
            new_i = np.clip(centers[i] + step * direction, -0.5, 0.5)
            old = centers[i].copy()
            centers[i] = new_i
            new_score = _score(centers)
            if new_score > current_score:
                current_score = new_score
                improved = True
            else:
                centers[i] = old
        
        if not improved:
            # Try smaller step
            step *= 0.7
            if step < 1e-6:
                break
        else:
            step = min(step * 1.05, 0.005)
    
    return centers, current_score

def _make_jittered_grid(seed):
    rng = np.random.RandomState(seed)
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    grid = grid[:N].copy()
    jitter = rng.randn(N, 3) * 0.015
    grid = np.clip(grid + jitter, -0.49, 0.49)
    return grid

def construct():
    t0 = time.time()
    deadline = t0 + 140.0
    
    best_centers = None
    best_score = -1.0
    
    for seed in range(30):
        if time.time() > deadline - 8.0:
            break
        
        centers = _make_jittered_grid(seed)
        
        # Repulsion phase
        dt = 0.02
        for step in range(1500):
            if time.time() > deadline - 10.0:
                break
            centers = _repulsion_step(centers, dt)
            dt = max(dt * 0.998, 0.002)
        
        # Local refinement
        centers, score = _local_refine(centers, max_iter=2000, time_limit=deadline - 5.0)
        
        if score > best_score:
            best_score = score
            best_centers = centers.copy()
        
        if best_score > 0.111:
            break
    
    if best_centers is None:
        m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        best_centers = grid[:N]
    
    return best_centers
# EVOLVE-BLOCK-END
