import numpy as np
import time
import scipy.spatial
import scipy.spatial.distance

def heuristics(distance_matrix):
    """
    Returns an edge-attractiveness matrix (larger is more attractive) for ACO-TSP.
    Uses a spectral graph-theoretic approach based on MST structure.
    """
    start_time = time.time()
    DEADLINE = 320.0  # Safety margin before 330s kill
    
    def elapsed():
        return time.time() - start_time
    
    n = distance_matrix.shape[0]
    dm = np.asarray(distance_matrix, dtype=np.float64)
    
    # Ensure diagonal is inf for MST computation
    dm_work = dm.copy()
    np.fill_diagonal(dm_work, np.inf)
    
    # 1. Compute MST using scipy
    try:
        from scipy.sparse.csgraph import minimum_spanning_tree
        mst = minimum_spanning_tree(dm_work)
        mst_dense = mst.toarray() if hasattr(mst, 'toarray') else np.asarray(mst)
    except Exception:
        # Fallback: Prim's algorithm
        mst_dense = np.zeros((n, n))
        in_mst = np.zeros(n, dtype=bool)
        in_mst[0] = True
        dist_from_mst = dm_work[0].copy()
        
        for _ in range(n - 1):
            # Find min edge from MST to non-MST nodes
            candidates = np.where(~in_mst)[0]
            if len(candidates) == 0:
                break
            min_idx = candidates[np.argmin(dist_from_mst[candidates])]
            in_mst[min_idx] = True
            mst_dense[min_idx, min_idx] = 0
            for j in range(n):
                if not in_mst[j]:
                    if dm_work[min_idx, j] < dist_from_mst[j]:
                        dist_from_mst[j] = dm_work[min_idx, j]
                    # Actually we need the edge weight, not just tracking
            # For MST edge tracking, we need to record which edge
            # Let's redo this more carefully
            break  # Fallback is complex, use scipy if available
        
        # Simpler fallback: just use distance-based heuristic
        if not np.any(mst_dense):
            # Use inverse distance squared
            d_safe = np.where(dm > 0, dm, 1.0)
            H = 1.0 / (d_safe ** 2.0)
            np.fill_diagonal(H, 0.0)
            H[~np.isfinite(H)] = 1e-9
            H[H <= 0] = 1e-9
            return H
    
    # 2. Compute node degrees in MST
    mst_binary = (mst_dense > 1e-9).astype(float)
    mst_binary[np.diag_indices(n)] = 0
    mst_degree = mst_binary.sum(axis=1)  # degree of each node in MST
    
    # 3. Compute MST cut property ranks
    # For each edge (i,j) in MST, its "rank" is related to how it connects components
    # We'll use: for each pair (i,j), define cut_rank based on MST path structure
    
    # Alternative: Use the MST edge weights to define a "structural importance"
    # For edges IN the MST: high attractiveness
    # For edges NOT in MST: lower attractiveness, scaled by distance
    
    # 4. Define heuristic based on:
    #    - Product of MST degrees (hubs connect strongly)
    #    - MST membership (boost MST edges)
    #    - Exponential decay of distance for non-MST edges
    
    # Base: inverse distance with sharpening
    d_safe = np.where(dm > 1e-10, dm, 1.0)
    base = 1.0 / (d_safe ** 1.5)  # Sharpened inverse distance
    
    # MST edge boost: MST edges get significant boost
    mst_edge_mask = (mst_dense > 1e-9)
    mst_boost = np.where(mst_edge_mask, 5.0, 1.0)
    
    # Degree-based boost: edges connecting high-degree nodes are more attractive
    # H[i,j] *= (deg[i] * deg[j])^alpha
    deg_product = np.outer(mst_degree, mst_degree)
    # Avoid zero degrees (isolated nodes in MST shouldn't happen for connected graph)
    deg_product = np.where(deg_product > 0, deg_product, 1.0)
    deg_boost = np.power(deg_product, 0.3)  # Sublinear degree boost
    
    # Combine: base * mst_boost * deg_boost
    H = base * mst_boost * deg_boost
    
    # 5. Spectral component: Use Laplacian eigenvalues to identify "backbone" edges
    # Compute graph Laplacian of the complete graph with weights 1/distance
    try:
        # Weighted adjacency
        W = 1.0 / d_safe
        np.fill_diagonal(W, 0.0)
        
        # Laplacian L = D - W
        D = np.diag(W.sum(axis=1))
        L = D - W
        
        # Compute smallest non-zero eigenvalues (Fiedler vector)
        # For performance, use partial eigendecomposition
        eigvals, eigvecs = np.linalg.eigh(L)
        
        # Fiedler vector (second smallest eigenvalue)
        # Identify edges where the Fiedler vector changes sign significantly
        fiedler = eigvecs[:, 1]  # Second eigenvector
        
        # Edge "cut strength" based on Fiedler vector difference
        # Edges that cross the "spectral cut" are important for tour structure
        fiedler_diff = np.abs(np.outer(fiedler, fiedler) - np.outer(fiedler, fiedler).T)
        # Actually: |fiedler[i] - fiedler[j]| for each edge
        fiedler_edge_diff = np.abs(fiedler[:, None] - fiedler[None, :])
        
        # Normalize
        fiedler_max = fiedler_edge_diff.max() + 1e-10
        fiedler_norm = fiedler_edge_diff / fiedler_max
        
        # Boost edges with high Fiedler difference (they are "cut" edges)
        # These are structurally important for separating the tour
        spectral_boost = 1.0 + 2.0 * fiedler_norm
    except Exception:
        spectral_boost = np.ones((n, n))
    
    H = H * spectral_boost
    
    # 6. Additional: Boost edges that are in the "2-tree" or near-MST structure
    # Compute second-nearest neighbors and boost those edges slightly
    # For each node, the 2nd and 3rd nearest neighbors
    k_nn = min(3, n - 1)
    if k_nn >= 2:
        sorted_dists = np.sort(dm, axis=1)
        # Distance to 2nd and 3rd nearest
        nn2_dist = sorted_dists[:, 1]  # 2nd nearest
        nn3_dist = sorted_dists[:, 2] if n > 3 else sorted_dists[:, 1]
        
        # Create boost for edges to 2nd/3rd nearest neighbors
        # This captures "almost-MST" edges
        # We'll use a smooth function based on relative distance
        max_nn2 = nn2_dist.max() + 1e-10
        
        # For each edge, if it's close to being a 2nd NN edge, boost it
        # Approximate: edges with distance close to 2nd NN distance
        # This is complex, so we'll use a simpler approach:
        # Boost edges that are within 1.2x of the 2nd NN distance for either endpoint
        nn2_boost = np.ones((n, n))
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                # Check if dm[i,j] is close to nn2_dist[i] or nn2_dist[j]
                if dm[i, j] <= nn2_dist[i] * 1.15 or (j < n and dm[i, j] <= nn3_dist[j] * 1.15):
                    nn2_boost[i, j] = 1.5
    
    H = H * nn2_boost
    
    # 7. Symmetrize
    H = 0.5 * (H + H.T)
    
    # 8. Clean up
    np.fill_diagonal(H, 0.0)
    H[~np.isfinite(H)] = 1e-9
    H[H <= 0] = 1e-9
    
    # 9. Deadline check
    if elapsed() > DEADLINE:
        return H
    
    return H
