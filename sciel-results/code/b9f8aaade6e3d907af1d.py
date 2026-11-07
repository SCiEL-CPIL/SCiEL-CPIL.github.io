"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Computes a spatially-smoothed, demand-aware heuristic matrix for ACO.
    
    Combines:
    1. Structural Inverse Distance: 1 / (d[i,j] + beta * d[j,0])
       - Encourages routes that consider the cost of returning to the depot.
       - beta is kept small (0.2) to provide a gentle bias without dominating.
    2. Demand Penalty: 1 / (1 + alpha * demand[j])
       - Discourages high-demand nodes to facilitate load balancing.
    3. Spatial Coherence: Boosts edges aligned with local neighborhood flow.
       - Uses k-NN principal direction to encourage smooth, non-crossing paths.
    """
    start_time = time.time()
    # Safety margin: 300s limit, kill at 330s. Heuristic calc is fast, but guard just in case.
    DEADLINE = start_time + 325.0

    d_mat = np.asarray(distance_matrix, dtype=np.float64)
    coords = np.asarray(coordinates, dtype=np.float64)
    dem = np.asarray(demands, dtype=np.float64)
    n = d_mat.shape[0]
    
    epsilon = 1e-9
    
    # 1. Structural Base Heuristic
    # Incorporates a mild return-to-depot cost.
    # d[j, 0] is distance from destination j to depot.
    depot_dist = d_mat[:, 0].copy()
    beta_struct = 0.2
    denom = d_mat + beta_struct * depot_dist[np.newaxis, :]
    
    # Avoid division by zero on diagonal
    denom = np.where(denom > epsilon, denom, epsilon)
    base = 1.0 / denom
    
    # 2. Demand Penalty
    # Prefer lower demand destinations to help fit more nodes per trip.
    # alpha=0.3 was effective in the parent program.
    alpha = 0.3
    demand_factor = 1.0 / (1.0 + alpha * dem)
    heuristic = base * demand_factor[np.newaxis, :]
    
    # 3. Spatial Coherence Bonus
    # Calculate local flow direction for each node using k-NN.
    if time.time() > DEADLINE:
        # Fallback if spatial calc is too slow (unlikely for N=50)
        heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
        return heuristic

    k = 5
    # Find k nearest neighbors for each node
    # We use argpartition for efficiency
    nn_indices = np.zeros((n, k), dtype=np.int64)
    for i in range(n):
        dists = d_mat[i].copy()
        dists[i] = np.inf
        # Ensure we pick k smallest
        nn_indices[i] = np.argpartition(dists, k)[:k]
    
    # Compute mean direction to k-nearest neighbors
    mean_dirs = np.zeros((n, 2), dtype=np.float64)
    for i in range(n):
        nn = nn_indices[i]
        vecs = coords[nn] - coords[i]
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms < epsilon] = epsilon
        vecs_norm = vecs / norms
        mean_dirs[i] = np.mean(vecs_norm, axis=0)
    
    # Normalize mean directions
    md_norms = np.linalg.norm(mean_dirs, axis=1, keepdims=True)
    md_norms[md_norms < epsilon] = epsilon
    mean_dirs_norm = mean_dirs / md_norms
    
    # Compute alignment of each edge (i, j) with the local flow of i
    # Vectorized calculation for alignment
    # edge_vecs[i, j] = coords[j] - coords[i]
    edge_vecs = coords[np.newaxis, :, :] - coords[:, np.newaxis, :]  # (n, n, 2)
    edge_norms = np.linalg.norm(edge_vecs, axis=2)  # (n, n)
    edge_norms_safe = np.where(edge_norms > epsilon, edge_norms, epsilon)
    edge_dirs = edge_vecs / edge_norms_safe[:, :, np.newaxis]  # (n, n, 2)
    
    # Alignment[i, j] = dot(mean_dirs_norm[i], edge_dirs[i, j])
    # mean_dirs_norm is (n, 2), edge_dirs is (n, n, 2)
    # We need to broadcast mean_dirs_norm to (n, 1, 2)
    alignment = np.sum(mean_dirs_norm[:, np.newaxis, :] * edge_dirs, axis=2)
    
    # Boost edges with positive alignment
    # beta_spatial controls the strength of the spatial boost
    beta_spatial = 0.5
    spatial_boost = 1.0 + beta_spatial * np.maximum(0.0, alignment)
    
    heuristic = heuristic * spatial_boost
    
    # 4. Final Cleanup
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
