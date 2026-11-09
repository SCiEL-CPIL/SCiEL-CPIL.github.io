"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Delaunay
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.sparse import csr_matrix
import time


def heuristics(distance_matrix):
    """
    Compute an edge-attractiveness matrix for ant colony optimization.
    
    Strategy: Combine multiple heuristic signals:
    1. Inverse distance (base)
    2. Delaunay triangulation edges get a boost
    3. MST edges get an additional boost
    4. Penalize very long edges relative to local neighborhood
    """
    n = distance_matrix.shape[0]
    start_time = time.time()
    
    # Base heuristic: inverse distance
    eps = 1e-9
    base = 1.0 / (distance_matrix + eps)
    
    # Extract coordinates from distance matrix (for 50 points in unit square)
    # We can recover approximate coordinates using MDS or just use the distance matrix directly
    # For Delaunay, we need coordinates. Let's use classical MDS to get 2D coords.
    
    # Classical MDS
    D = distance_matrix.copy()
    D = np.where(D > 0, D, 0)  # zero out self-distances
    
    # Double centering
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ (D ** 2) @ J
    
    # Eigen decomposition
    eigvals, eigvecs = np.linalg.eigh(B)
    
    # Take top 2 eigenvectors (largest eigenvalues)
    # Sort by eigenvalue descending
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    
    # Ensure real and positive eigenvalues
    eigvals = np.maximum(eigvals, 0)
    coords = eigvecs[:, :2] * np.sqrt(eigvals[:2])
    
    # Now we have approximate 2D coordinates
    # Compute Delaunay triangulation
    try:
        tri = Delaunay(coords)
        delaunay_edges = set()
        for simplex in tri.simplices:
            i, j, k = simplex
            delaunay_edges.add((min(i, j), max(i, j)))
            delaunay_edges.add((min(i, k), max(i, k)))
            delaunay_edges.add((min(j, k), max(j, k)))
    except Exception:
        delaunay_edges = set()
    
    # Compute MST
    try:
        # Use distance matrix for MST
        mst = minimum_spanning_tree(csr_matrix(distance_matrix))
        mst_dense = mst.toarray()
        mst_edges = set()
        for i in range(n):
            for j in range(n):
                if mst_dense[i, j] > 0:
                    mst_edges.add((min(i, j), max(i, j)))
    except Exception:
        mst_edges = set()
    
    # Build heuristic matrix
    heuristic = base.copy()
    
    # Boost Delaunay edges
    delaunay_boost = 2.0
    for i, j in delaunay_edges:
        heuristic[i, j] *= delaunay_boost
        heuristic[j, i] *= delaunay_boost
    
    # Boost MST edges further
    mst_boost = 1.5
    for i, j in mst_edges:
        heuristic[i, j] *= mst_boost
        heuristic[j, i] *= mst_boost
    
    # Penalize long edges: edges that are much longer than the median edge length
    # Get all unique edge lengths
    triu_indices = np.triu_indices(n, k=1)
    edge_lengths = distance_matrix[triu_indices]
    median_len = np.median(edge_lengths)
    
    # For edges significantly longer than median, reduce attractiveness
    # This encourages local movement
    for i in range(n):
        for j in range(i + 1, n):
            d = distance_matrix[i, j]
            if d > 2.0 * median_len:
                # Reduce heuristic for very long edges
                penalty = median_len / d
                heuristic[i, j] *= penalty
                heuristic[j, i] *= penalty
    
    # Ensure all values are positive and finite
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > 280:  # Safety margin before 300s limit
        # Return a simpler heuristic if we're running out of time
        return 1.0 / (distance_matrix + eps)
    
    return heuristic
# EVOLVE-BLOCK-END
