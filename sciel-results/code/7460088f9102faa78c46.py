"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14
D = 3

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

# Deadline guard: keep a safety margin before the 150s wall-clock limit
_START_TIME = time.time()
TIME_LIMIT = 145.0  # seconds, leave 5s safety margin

def _time_remaining():
    return TIME_LIMIT - (time.time() - _START_TIME)

def _pairwise_distances(points):
    """Compute all pairwise squared distances for n x d array."""
    # points: (n, d)
    # squared distances: (n, n)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    sq_dists = np.sum(diff ** 2, axis=-1)
    return sq_dists

def _score(points):
    """Compute (max dist / min dist)^2. Returns inf if any points coincide."""
    sq_dists = _pairwise_distances(points)
    # Exclude self-distances (diagonal)
    mask = ~np.eye(len(points), dtype=bool)
    sq_dists_masked = sq_dists[mask]
    if len(sq_dists_masked) == 0:
        return np.inf
    min_sq = np.min(sq_dists_masked)
    max_sq = np.max(sq_dists_masked)
    if min_sq < 1e-30:
        return np.inf
    return max_sq / min_sq

def _objective(params):
    """Objective: maximize log(min_sq) - log(max_sq) to minimize the ratio.
    We minimize negative of that, i.e., log(max_sq) - log(min_sq).
    """
    points = params.reshape(N, D)
    sq_dists = _pairwise_distances(points)
    mask = ~np.eye(N, dtype=bool)
    sq_dists_masked = sq_dists[mask]
    
    min_sq = np.min(sq_dists_masked)
    max_sq = np.max(sq_dists_masked)
    
    if min_sq < 1e-30 or max_sq < 1e-30:
        return 1e10
    
    # We want to minimize log(max_sq) - log(min_sq) = log(max_sq/min_sq)
    # This is equivalent to minimizing the log of the ratio
    return np.log(max_sq) - np.log(min_sq)

def _objective_with_penalty(params, penalty_weight=1000.0):
    """Objective with penalty for near-coincident points."""
    points = params.reshape(N, D)
    sq_dists = _pairwise_distances(points)
    mask = ~np.eye(N, dtype=bool)
    sq_dists_masked = sq_dists[mask]
    
    min_sq = np.min(sq_dists_masked)
    max_sq = np.max(sq_dists_masked)
    
    if min_sq < 1e-30:
        return 1e10
    
    base_obj = np.log(max_sq) - np.log(min_sq)
    
    # Penalty: encourage all distances to be well-separated
    # Penalize very small distances
    penalty = 0.0
    for d in sq_dists_masked:
        if d < 1e-6:
            penalty += penalty_weight / (d + 1e-12)
    
    return base_obj + penalty

def _center_and_scale(points):
    """Center points at origin and scale to unit norm."""
    centroid = np.mean(points, axis=0)
    centered = points - centroid
    norm = np.linalg.norm(centered)
    if norm > 1e-15:
        centered = centered / norm
    return centered

def _initial_tetrahedron_with_midpoints():
    """Create 4 vertices of regular tetrahedron + 6 edge midpoints + 4 face centroids = 14 points."""
    # Regular tetrahedron vertices
    a = np.sqrt(2.0 / 3.0)
    v1 = np.array([a, a, a])
    v2 = np.array([a, -a, -a])
    v3 = np.array([-a, a, -a])
    v4 = np.array([-a, -a, a])
    
    vertices = np.array([v1, v2, v3, v4])
    
    # 6 edge midpoints
    edges = [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]
    midpoints = np.array([(vertices[i] + vertices[j]) / 2.0 for i, j in edges])
    
    # 4 face centroids (centroids of the 4 triangular faces)
    faces = [(0,1,2), (0,1,3), (0,2,3), (1,2,3)]
    centroids = np.array([(vertices[i] + vertices[j] + vertices[k]) / 3.0 for i, j, k in faces])
    
    points = np.vstack([vertices, midpoints, centroids])
    return _center_and_scale(points)

def _initial_random_on_sphere(seed=None):
    """Generate random points on unit sphere."""
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random
    
    points = rng.randn(N, D)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    points = points / norms
    return points

def _optimize_from_initial(init_points, max_iter=2000, tol=1e-12):
    """Run Powell optimization from a given initial configuration."""
    init_flat = init_points.flatten()
    
    # Try Powell first (derivative-free, good for non-smooth)
    result = minimize(
        _objective,
        init_flat,
        method='Powell',
        options={'maxiter': max_iter, 'ftol': tol, 'xtol': tol}
    )
    
    best_points = result.x.reshape(N, D)
    best_score = _score(best_points)
    
    return best_points, best_score

def construct():
    best_points = None
    best_score = np.inf
    
    # Try multiple initial configurations
    initial_configs = []
    
    # 1. Tetrahedron + midpoints + centroids
    initial_configs.append(_initial_tetrahedron_with_midpoints())
    
    # 2-5. Random sphere configurations with different seeds
    for seed in [42, 123, 456, 789]:
        initial_configs.append(_initial_random_on_sphere(seed))
    
    for i, init in enumerate(initial_configs):
        if _time_remaining() < 10.0:
            break
        
        try:
            points, score = _optimize_from_initial(init, max_iter=500, tol=1e-10)
            if score < best_score:
                best_score = score
                best_points = points.copy()
        except Exception:
            continue
    
    # Final polish: run a tighter optimization from the best found
    if best_points is not None and _time_remaining() > 15.0:
        try:
            points, score = _optimize_from_initial(best_points, max_iter=1000, tol=1e-14)
            if score < best_score:
                best_score = score
                best_points = points.copy()
        except Exception:
            pass
    
    if best_points is None:
        # Fallback: return the tetrahedron-based configuration
        best_points = _initial_tetrahedron_with_midpoints()
    
    return best_points
# EVOLVE-BLOCK-END
