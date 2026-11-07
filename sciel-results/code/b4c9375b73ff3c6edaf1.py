"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import Delaunay


def heuristics(distance_matrix):
    start_time = time.time()
    deadline = 300.0  # seconds, with safety margin
    
    n = distance_matrix.shape[0]
    if n < 3:
        return 1.0 / (distance_matrix + 1e-9)
    
    # Extract points from distance matrix (reconstruct positions)
    # Use classical MDS to get 2D coordinates
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
    
    # Also build k-NN graph for additional sparse structure
    # Use k=8 nearest neighbors
    k = min(8, n - 1)
    knn_adj = np.zeros((n, n), dtype=bool)
    for i in range(n):
        dists = distance_matrix[i].copy()
        dists[i] = np.inf
        neighbors = np.argsort(dists)[:k]
        for j in neighbors:
            knn_adj[i, j] = True
            knn_adj[j, i] = True
    
    # Combine Delaunay and k-NN: edges in either graph are "good"
    good_edges = delaunay_adj | knn_adj
    
    # Create heuristic: high for good edges, low for others
    base_heur = 1.0 / (distance_matrix + 1e-9)
    
    # Scale: good edges get multiplied by a large factor
    # Non-good edges get multiplied by a small factor
    boost = 100.0
    damp = 1e-6
    
    heur = np.where(good_edges, base_heur * boost, base_heur * damp)
    
    # Ensure diagonal is 0 (no self-loops)
    np.fill_diagonal(heur, 0.0)
    
    # Ensure all values are positive and finite
    heur = np.where(np.isfinite(heur) & (heur > 0), heur, 1e-9)
    
    # Check deadline
    elapsed = time.time() - start_time
    if elapsed > deadline - 5.0:
        # Return a simple fallback if we're running out of time
        return 1.0 / (distance_matrix + 1e-9)
    
    return heur
# EVOLVE-BLOCK-END
