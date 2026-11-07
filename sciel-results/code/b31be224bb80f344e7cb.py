"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import sys


def _compute_mst_edges(d, n):
    """
    Compute MST edges using Prim's algorithm (dense graph version).
    Returns list of (u, v) edges in the MST.
    """
    in_mst = np.zeros(n, dtype=bool)
    min_dist = np.full(n, np.inf)
    parent = np.full(n, -1, dtype=np.int64)
    
    min_dist[0] = 0.0
    mst_edges = []
    
    for _ in range(n):
        # Find the node with minimum distance not in MST
        u = -1
        best = np.inf
        for i in range(n):
            if not in_mst[i] and min_dist[i] < best:
                best = min_dist[i]
                u = i
        
        if u == -1:
            break
            
        in_mst[u] = True
        if parent[u] != -1:
            mst_edges.append((parent[u], u))
        
        # Update distances
        for v in range(n):
            if not in_mst[v] and d[u, v] < min_dist[v]:
                min_dist[v] = d[u, v]
                parent[v] = u
    
    return mst_edges


def heuristics(distance_matrix):
    start_time = time.time()
    deadline = start_time + 280.0  # Safety margin before 330s limit
    
    d = np.asarray(distance_matrix, dtype=np.float64)
    n = d.shape[0]
    
    # Ensure symmetry and handle diagonal
    d = (d + d.T) / 2.0
    np.fill_diagonal(d, 0.0)
    
    # Compute MST
    mst_edges = _compute_mst_edges(d, n)
    
    # Build MST adjacency and degrees
    mst_adj = np.zeros((n, n), dtype=np.float64)
    mst_degree = np.zeros(n, dtype=np.int64)
    for u, v in mst_edges:
        mst_adj[u, v] = 1.0
        mst_adj[v, u] = 1.0
        mst_degree[u] += 1
        mst_degree[v] += 1
    
    # Compute edge ranks in MST cut property
    # For each edge, compute a rank based on how many MST edges it "cuts"
    # We use a spectral approach: compute the Laplacian of the MST and use
    # the eigenvectors to measure connectivity
    
    # Build MST Laplacian
    L = np.zeros((n, n), dtype=np.float64)
    for u, v in mst_edges:
        L[u, u] += 1
        L[v, v] += 1
        L[u, v] -= 1
        L[v, u] -= 1
    
    # Compute eigenvalues and eigenvectors of the Laplacian
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(L)
        # Sort by eigenvalue
        idx = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Use the Fiedler vector (second smallest eigenvalue's eigenvector)
        # to compute a "spectral rank" for edges
        fiedler = eigenvectors[:, 1] if n > 1 else np.ones(n)
        
        # Edge spectral rank: difference in Fiedler values
        # Edges that connect nodes with similar Fiedler values are "internal"
        # Edges that connect nodes with different Fiedler values are "boundary"
        spectral_rank = np.abs(fiedler[:, np.newaxis] - fiedler[np.newaxis, :])
        
    except Exception:
        # Fallback: use distance-based rank
        spectral_rank = d.copy()
    
    # Normalize spectral rank to [0, 1]
    sr_min = spectral_rank.min()
    sr_max = spectral_rank.max()
    if sr_max > sr_min:
        spectral_rank = (spectral_rank - sr_min) / (sr_max - sr_min)
    else:
        spectral_rank = np.zeros_like(spectral_rank)
    
    # Compute heuristic as product of:
    # 1. Degree boost: higher degree nodes in MST are more important
    # 2. Exponential decay of spectral rank: edges with low spectral rank (internal) are more attractive
    # 3. Inverse distance: standard ACO heuristic
    
    # Degree boost: sqrt of product of degrees (geometric mean)
    degree_boost = np.sqrt(np.outer(mst_degree.astype(np.float64), mst_degree.astype(np.float64)))
    # Normalize degree boost
    db_min = degree_boost.min()
    db_max = degree_boost.max()
    if db_max > db_min:
        degree_boost = (degree_boost - db_min) / (db_max - db_min) + 0.5
    else:
        degree_boost = np.ones_like(degree_boost)
    
    # Exponential decay of spectral rank
    # Low spectral rank (0) -> high attractiveness, high spectral rank (1) -> low attractiveness
    exp_decay = np.exp(-2.0 * spectral_rank)
    
    # Inverse distance with moderate power
    eta_distance = 1.0 / (np.power(d + 1e-9, 1.2))
    
    # Combine all components
    # Weighted combination: emphasize MST structure and spectral properties
    eta = eta_distance * (0.5 + 0.5 * degree_boost) * exp_decay
    
    # Ensure diagonal is handled properly
    np.fill_diagonal(eta, 0.0)
    
    # Ensure all values are positive and finite
    eta[~np.isfinite(eta)] = 1e-9
    eta[eta <= 0] = 1e-9
    
    # Check time
    if time.time() > deadline:
        return eta
    
    return eta
# EVOLVE-BLOCK-END
