"""Heuristic matrix for ant colony optimization of TSP."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def heuristics(distance_matrix):
    """
    Returns an edge-attractiveness matrix for ACO.
    
    Strategy: Geometric Embedding (MDS) + Local Structure.
    1. Reconstruct 2D points from the distance matrix using Classical MDS.
    2. Identify 'safe' edges: those on the convex hull and k-nearest neighbors.
    3. Construct heuristic as inverse distance, heavily penalizing edges that 
       are not part of the local geometric structure (long chords, crossing risks).
    This guides ants to follow the geometric perimeter and local paths, avoiding 
    the 'global search' noise of consensus methods which can be slow or noisy 
    for small n=50.
    """
    start_time = time.time()
    deadline = start_time + 290  # Safety margin
    
    dm = np.asarray(distance_matrix, dtype=float)
    n = dm.shape[0]
    
    if n == 0:
        return dm.copy()
    
    # 1. Reconstruct coordinates using Classical MDS
    # D is the distance matrix. We want X such that ||Xi - Xj|| ~ D[i,j]
    # Step 1: Double centering
    # J = I - (1/n) * 11^T
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ (dm ** 2) @ J
    
    # Step 2: Eigenvalue decomposition
    # B is symmetric, so eigenvalues are real
    eigvals, eigvecs = np.linalg.eigh(B)
    
    # Sort by descending eigenvalue
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    
    # Take top 2 dimensions (since points are in unit square)
    # Ensure eigenvalues are non-negative (numerical stability)
    eigvals[:2] = np.maximum(eigvals[:2], 0)
    
    # Coordinates
    U = eigvecs[:, :2]
    S = np.sqrt(eigvals[:2])
    X = U * S
    
    # 2. Identify local structure
    # A. Convex Hull: Edges on the hull are very likely to be in the optimal TSP tour
    #    (for points in convex position, the optimal tour is the hull).
    #    Even for non-convex, hull edges are almost always part of the optimal tour.
    try:
        from scipy.spatial import ConvexHull
        hull = ConvexHull(X)
        hull_edges = set()
        for simplex in hull.simplices:
            hull_edges.add((simplex[0], simplex[1]))
            hull_edges.add((simplex[1], simplex[0]))
    except Exception:
        hull_edges = set()
    
    # B. K-Nearest Neighbors: The optimal TSP tour for random points in a square
    #    typically consists of edges that are among the nearest neighbors (k=4 or 6 is safe).
    #    We mark edges to the 4 nearest neighbors as "local".
    k = 4
    local_edges = set()
    # For each point, find k nearest neighbors
    # dm is symmetric, so we can just look at rows
    for i in range(n):
        # Get distances from i to all others
        dists = dm[i].copy()
        dists[i] = np.inf # Exclude self
        # Get indices of k smallest
        nn_indices = np.argpartition(dists, k)[:k]
        for j in nn_indices:
            local_edges.add((i, j))
            local_edges.add((j, i))
    
    # Combine "good" edges
    good_edges = hull_edges.union(local_edges)
    
    # 3. Construct Heuristic
    # Base: Inverse distance
    # Penalty: If edge is NOT in good_edges, multiply by a small factor.
    # This creates a "funnel" that strongly encourages ants to stay on the 
    # geometrically plausible path.
    
    # Inverse distance
    inv_dist = 1.0 / (dm + 1e-9)
    
    # Mask for good edges
    # Create a matrix where 1.0 is good, 0.1 is bad (penalty)
    # We want to strongly suppress non-local edges.
    # ACO uses eta[i,j] * tau[i,j]. If eta is small, the ant is less likely to pick it.
    # Let's set penalty to 0.05 for non-local, 1.0 for local.
    # Actually, to be safe and ensure positive values, we can do:
    # eta = inv_dist * (1.0 if good else 0.1)
    
    heuristic = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            if i == j:
                heuristic[i, j] = 0.0
            else:
                if (i, j) in good_edges:
                    heuristic[i, j] = inv_dist[i, j]
                else:
                    # Strong penalty for non-local edges
                    # This forces the ants to explore the "obvious" geometric structure first
                    heuristic[i, j] = inv_dist[i, j] * 0.05
    
    # Ensure no values are not positive or non-finite
    # The problem states: "Heuristic entries that are not positive become 1e-9"
    # Our heuristic is always positive (inv_dist > 0, multiplier > 0).
    # But let's be safe.
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    # Diagonal should probably be 0 or very small, but ACO usually ignores i==j.
    # The contract says "edge-attractiveness matrix... of the same shape".
    # Usually diagonal is irrelevant if code checks i!=j, but let's set to 1e-9 to be safe.
    np.fill_diagonal(heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
