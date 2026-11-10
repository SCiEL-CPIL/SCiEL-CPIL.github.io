"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import cKDTree


def _compute_score(points):
    """Compute the score: min(min pairwise dist / 2, min (1 - |c_i|))."""
    if len(points) < 2:
        return 0.0
    tree = cKDTree(points)
    # Find minimum pairwise distance
    pairs = tree.query_pairs(1e-10)
    if not pairs:
        min_dist = float('inf')
    else:
        # Sample if too many pairs
        if len(pairs) > 10000:
            sample = np.random.choice(len(pairs), 10000, replace=False)
            sampled_pairs = [pairs[i] for i in sample]
            min_dist = min(np.linalg.norm(points[i] - points[j]) for i, j in sampled_pairs)
        else:
            min_dist = min(np.linalg.norm(points[i] - points[j]) for i, j in pairs)
    # Minimum distance to boundary
    radii = np.linalg.norm(points, axis=1)
    min_boundary = 1.0 - np.max(radii)
    return min(min_dist / 2.0, min_boundary)


def _local_optimize(points, deadline, max_steps=5000):
    """Local optimization using greedy pairwise repulsion."""
    n = len(points)
    current_score = _compute_score(points)
    best_points = points.copy()
    best_score = current_score
    
    # Find initial minimum distance
    tree = cKDTree(points)
    min_dist, _ = tree.query(points, k=2)
    min_dist = min_dist[:, 1].min()
    
    step_size = min_dist * 0.1
    rng = np.random.default_rng(42)
    
    for step in range(max_steps):
        if time.time() > deadline:
            break
        
        # Find the pair with minimum distance
        tree = cKDTree(points)
        dists, indices = tree.query(points, k=2)
        min_idx = np.argmin(dists[:, 1])
        min_d = dists[min_idx, 1]
        
        # Move points in this pair apart
        i, j = min_idx, indices[min_idx, 1]
        direction = points[i] - points[j]
        norm = np.linalg.norm(direction)
        if norm < 1e-12:
            direction = rng.normal(size=2)
            norm = np.linalg.norm(direction)
        direction = direction / norm
        
        # Move each point by half the step in opposite directions
        delta = step_size / 2.0 * direction
        points[i] += delta
        points[j] -= delta
        
        # Clamp to unit circle
        for idx in [i, j]:
            r = np.linalg.norm(points[idx])
            if r > 1.0:
                points[idx] = points[idx] / r * 0.999
        
        # Check if score improved
        new_score = _compute_score(points)
        if new_score > best_score:
            best_score = new_score
            best_points = points.copy()
            # Decrease step size slightly
            step_size *= 0.99
        else:
            # Revert
            points[i] -= delta
            points[j] += delta
            step_size *= 0.95
        
        # Occasionally try random perturbations
        if step % 100 == 0:
            idx = rng.integers(n)
            perturbation = rng.normal(scale=step_size, size=2)
            old = points[idx].copy()
            points[idx] += perturbation
            r = np.linalg.norm(points[idx])
            if r > 1.0:
                points[idx] = points[idx] / r * 0.999
            new_score = _compute_score(points)
            if new_score > best_score:
                best_score = new_score
                best_points = points.copy()
            else:
                points[idx] = old
    
    return best_points, best_score


def construct():
    # Hexagonal lattice cut to the unit disk
    spacing = 2.0 / (np.sqrt(N) + 1.0)
    while True:
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        points, radius = points[radius <= 1.0 - spacing / 2.0], radius[radius <= 1.0 - spacing / 2.0]
        if len(points) >= N:
            points = points[np.argsort(radius, kind="stable")[:N]]
            break
        spacing *= 0.98
    
    # Deadline: 110s to leave safety margin
    deadline = time.time() + 110.0
    
    # Local optimization
    points, score = _local_optimize(points, deadline)
    
    return points
# EVOLVE-BLOCK-END
