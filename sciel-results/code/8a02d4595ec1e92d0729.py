import numpy as np
import time
from scipy.spatial import Delaunay

_T0 = time.time()
_BUDGET = 330.0
_DEADLINE = _T0 + _BUDGET
_TIME_MARGIN = 15.0

def _remaining():
    return _DEADLINE - _TIME_MARGIN - time.time()

def _safe_heuristic(distance_matrix, coordinates):
    """Fallback heuristic: pure inverse distance."""
    dm = np.asarray(distance_matrix, dtype=float)
    n = dm.shape[0]
    d = dm.copy()
    np.fill_diagonal(d, 1e9)
    h = 1.0 / (d + 1e-9)
    np.fill_diagonal(h, 0.0)
    h[~np.isfinite(h)] = 1e-9
    h[h <= 0] = 1e-9
    return h

def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Compute an edge-attractiveness matrix for ACO on CVRP.
    
    Strategy:
    - Uses Delaunay Triangulation to identify geometrically proximal neighbors.
    - Edges in the Delaunay graph get high weights; others get near-zero.
    - This enforces a planar constraint, preventing crossing routes.
    - Combines with inverse distance for finer granularity among valid edges.
    """
    dm = np.asarray(distance_matrix, dtype=float)
    coords = np.asarray(coordinates, dtype=float)
    n = dm.shape[0]
    
    # Basic validation
    if dm.shape != (n, n) or not np.isfinite(dm).all() or coords.shape[0] != n:
        return _safe_heuristic(dm, coords)
    
    if _remaining() < 5.0:
        return _safe_heuristic(dm, coords)
    
    eps = 1e-9
    
    # Compute inverse distance matrix (sharp exponent)
    d_safe = dm.copy()
    np.fill_diagonal(d_safe, 1e9)
    inv_dist = 1.0 / (d_safe ** 1.5)
    
    # Initialize heuristic with small base value
    h = np.full((n, n), 1e-9)
    
    # Get Delaunay triangulation of all points (including depot)
    try:
        if n >= 3:
            tri = Delaunay(coords)
            # Build adjacency from triangulation
            # For each simplex, connect all pairs
            for simplex in tri.simplices:
                for i in range(len(simplex)):
                    for j in range(i + 1, len(simplex)):
                        a, b = simplex[i], simplex[j]
                        if a != b:
                            # Boost edges in the Delaunay graph
                            h[a, b] = inv_dist[a, b]
                            h[b, a] = inv_dist[b, a]
    except Exception:
        # Fallback: use k-nearest neighbors
        try:
            from scipy.spatial import cKDTree
            tree = cKDTree(coords)
            k = min(6, n - 1)
            dists, indices = tree.query(coords, k=k + 1)  # +1 for self
            for i in range(n):
                for j_idx in range(1, k + 1):
                    j = indices[i, j_idx]
                    if i != j:
                        h[i, j] = inv_dist[i, j]
                        h[j, i] = inv_dist[j, i]
        except Exception:
            # Ultimate fallback
            h = inv_dist.copy()
            np.fill_diagonal(h, 0.0)
    
    # Ensure diagonal is zero
    np.fill_diagonal(h, 0.0)
    
    # Handle non-finite or non-positive values
    h[~np.isfinite(h)] = 1e-9
    h[h <= 0] = 1e-9
    
    # Normalize to [0, 1] range
    max_h = np.max(h)
    if max_h > 0:
        h = h / max_h
    
    # Re-apply positivity constraint
    h[h <= 0] = 1e-9
    np.fill_diagonal(h, 0.0)
    
    return h
