# EVOLVE-BLOCK-START
import numpy as np
import itertools
from scipy.spatial.distance import pdist
import time


def _compute_metric(points):
    """Compute (d_min/d_max)^2 for a set of points."""
    if len(points) < 2:
        return 0.0
    d = pdist(points)
    d_min = np.min(d)
    d_max = np.max(d)
    if d_max == 0:
        return 0.0
    return (d_min / d_max) ** 2


def _generate_dodecahedron_vertices():
    """Generate the 20 vertices of a regular dodecahedron."""
    phi = (1 + np.sqrt(5)) / 2.0
    verts = []
    # (±1, ±1, ±1)
    for x in [-1, 1]:
        for y in [-1, 1]:
            for z in [-1, 1]:
                verts.append([x, y, z])
    # (0, ±1/φ, ±φ), (±1/φ, ±φ, 0), (±φ, 0, ±1/φ)
    for a in [-1/phi, 1/phi]:
        for b in [-phi, phi]:
            verts.append([0, a, b])
            verts.append([a, b, 0])
            verts.append([b, 0, a])
    verts = np.array(verts, dtype=np.float64)
    # Normalize to unit sphere
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts = verts / norms
    return verts


def _select_best_subset(verts, k, max_combos=None):
    """Find the best k-subset of vertices by metric."""
    n = len(verts)
    best_score = -1.0
    best_subset = None
    
    if max_combos is None:
        max_combos = len(list(itertools.combinations(range(n), k)))
    
    count = 0
    for combo in itertools.combinations(range(n), k):
        subset = verts[list(combo)]
        score = _compute_metric(subset)
        if score > best_score:
            best_score = score
            best_subset = subset.copy()
        count += 1
        if count >= max_combos:
            break
    
    return best_subset, best_score


def _local_optimize(points, max_iter=5000, tol=1e-12):
    """Local optimization to maximize (d_min/d_max)^2 using gradient-free approach."""
    n = len(points)
    best_points = points.copy()
    best_score = _compute_metric(best_points)
    
    # Use a simple iterative improvement approach
    step_size = 0.01
    for iteration in range(max_iter):
        improved = False
        for i in range(n):
            for dim in range(3):
                # Try small perturbations in each direction
                for delta in [step_size, -step_size]:
                    trial = best_points.copy()
                    trial[i, dim] += delta
                    score = _compute_metric(trial)
                    if score > best_score + 1e-15:
                        best_points = trial
                        best_score = score
                        improved = True
                        break
        
        if not improved:
            # Decrease step size
            step_size *= 0.5
            if step_size < 1e-10:
                break
    
    return best_points, best_score


def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions to maximize the ratio of minimum to maximum distance.
    
    Returns
        points: np.ndarray of shape (14,3) containing the coordinates of the 14 points.
    """
    start_time = time.time()
    deadline = start_time + 600  # Safety margin before 630s limit
    
    n = 14
    d = 3
    
    # Step 1: Generate dodecahedron vertices
    verts = _generate_dodecahedron_vertices()
    
    # Step 2: Find best 14-subset from dodecahedron vertices
    best_subset, best_score = _select_best_subset(verts, n)
    
    # Check time
    if time.time() > deadline - 30:
        return best_subset
    
    # Step 3: Local optimization
    optimized_points, optimized_score = _local_optimize(best_subset, max_iter=2000)
    
    # Check time
    if time.time() > deadline - 15:
        return best_subset if best_score >= optimized_score else optimized_points
    
    # Return the better of the two
    if optimized_score >= best_score:
        return optimized_points
    else:
        return best_subset


# EVOLVE-BLOCK-END
