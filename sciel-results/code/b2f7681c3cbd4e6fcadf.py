"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Delaunay, distance_matrix as sp_dist


def heuristics(distance_matrix):
    n = distance_matrix.shape[0]
    if n < 3:
        return 1.0 / (distance_matrix + 1e-9)
    
    # Extract points from distance matrix (reconstruct positions)
    # Since we have distance matrix, we need to embed in 2D
    # Use classical MDS or just assume the distance matrix comes from 2D points
    # Actually, for 50 points in unit square, we can try to recover coordinates
    # But simpler: use the distance matrix directly for Delaunay via cKDTree won't work
    # We need actual coordinates. Let's use MDS to get 2D coordinates.
    
    from scipy.spatial.distance import squareform
    import numpy.linalg as la
    
    # Classical MDS to recover 2D coordinates
    # B = -0.5 * J * D^2 * J where J is centering matrix
    D2 = distance_matrix ** 2
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ D2 @ J
    
    # Eigenvalue decomposition
    eigvals, eigvecs = la.eigh(B)
    # Take top 2 eigenvalues
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    
    # Coordinates in 2D
    coords = eigvecs[:, :2] * np.sqrt(np.maximum(eigvals[:2], 0))
    
    # Compute Delaunay triangulation
    try:
        tri = Delaunay(coords)
    except Exception:
        # Fallback to inverse distance
        return 1.0 / (distance_matrix + 1e-9)
    
    # Build adjacency from Delaunay triangulation
    delaunay_adj = np.zeros((n, n), dtype=bool)
    for simplex in tri.simplices:
        for i in range(3):
            for j in range(i + 1, 3):
                u, v = simplex[i], simplex[j]
                delaunay_adj[u, v] = True
                delaunay_adj[v, u] = True
    
    # Create heuristic: high for Delaunay edges, low for others
    # For Delaunay edges, use inverse distance with a boost
    # For non-Delaunay edges, use very low value
    
    base_heur = 1.0 / (distance_matrix + 1e-9)
    
    # Scale: Delaunay edges get multiplied by a large factor
    # Non-Delaunay edges get multiplied by a small factor
    boost = 100.0
    damp = 1e-6
    
    heur = np.where(delaunay_adj, base_heur * boost, base_heur * damp)
    
    # Ensure diagonal is 0 (no self-loops)
    np.fill_diagonal(heur, 0.0)
    
    # Ensure all values are positive and finite
    heur = np.where(np.isfinite(heur) & (heur > 0), heur, 1e-9)
    
    return heur
# EVOLVE-BLOCK-END
