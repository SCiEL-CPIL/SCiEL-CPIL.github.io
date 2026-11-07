# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import Delaunay


def heuristics(distance_matrix):
    t0 = time.time()
    n = distance_matrix.shape[0]
    
    d = distance_matrix.astype(np.float64).copy()
    np.fill_diagonal(d, np.inf)
    
    # Reconstruct 2D points via classical MDS
    D2 = d ** 2
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ D2 @ J
    eigvals, eigvecs = np.linalg.eigh(B)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    eigvals = np.clip(eigvals, 0, None)
    coords = eigvecs[:, :2] * np.sqrt(eigvals[:2])
    
    # Compute Delaunay triangulation - edges here contain optimal TSP tour
    try:
        tri = Delaunay(coords)
        delaunay_edges = set()
        for simplex in tri.simplices:
            for i in range(3):
                for j in range(i+1, 3):
                    a, b = simplex[i], simplex[j]
                    delaunay_edges.add((min(a,b), max(a,b)))
    except Exception:
        delaunay_edges = set()
    
    # Also compute k-nearest neighbors as fallback/supplement
    k = 4
    nn_indices = np.argpartition(d, k, axis=1)[:, :k]
    
    # Build edge attractiveness matrix
    result = np.ones((n, n), dtype=np.float64)
    
    # Base: inverse distance squared (smooth ranking)
    eps = 1e-9
    d_safe = d + eps
    base = 1.0 / (d_safe ** 2.0)
    result = base.copy()
    
    # Strong boost for Delaunay edges (guaranteed to contain optimal tour)
    for (a, b) in delaunay_edges:
        result[a, b] *= 15.0
        result[b, a] *= 15.0
    
    # Additional boost for mutual k-nearest neighbors
    for i in range(n):
        for j in nn_indices[i]:
            if j in nn_indices[i].tolist() and i in np.argpartition(d[j], k)[:k]:
                result[i, j] *= 3.0
                result[j, i] *= 3.0
    
    # Symmetrize
    result = 0.5 * (result + result.T)
    
    # Ensure positive and finite
    result = np.where(np.isfinite(result) & (result > 0), result, 1e-9)
    
    return result
# EVOLVE-BLOCK-END
