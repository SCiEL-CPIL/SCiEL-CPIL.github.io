# EVOLVE-BLOCK-START
import numpy as np


def heuristics(distance_matrix):
    n = distance_matrix.shape[0]
    if n <= 1:
        return np.ones_like(distance_matrix, dtype=float)
    
    # Compute MST using Prim's algorithm (efficient for dense graphs with O(n^2))
    # We'll use a simple O(n^2) Prim's algorithm since n=50 is small
    in_mst = np.zeros(n, dtype=bool)
    in_mst[0] = True
    min_dist = distance_matrix[0].copy()
    
    mst_adj = np.zeros((n, n), dtype=bool)
    
    for _ in range(n - 1):
        # Find the node with minimum distance to the MST
        # Mask out nodes already in MST
        min_dist_masked = min_dist.copy()
        min_dist_masked[in_mst] = np.inf
        u = np.argmin(min_dist_masked)
        
        in_mst[u] = True
        # The edge connecting u to the MST: find which node in MST has min distance
        # Actually, we need to track the parent. Let's redo with parent tracking.
        
        # Update min_dist
        min_dist = np.minimum(min_dist, distance_matrix[u])
        min_dist[in_mst] = np.inf
    
    # The above doesn't track edges. Let's redo properly with parent tracking.
    in_mst = np.zeros(n, dtype=bool)
    in_mst[0] = True
    min_dist = distance_matrix[0].copy()
    parent = np.full(n, -1, dtype=int)
    parent[0] = -1
    
    for _ in range(n - 1):
        min_dist_masked = min_dist.copy()
        min_dist_masked[in_mst] = np.inf
        u = np.argmin(min_dist_masked)
        
        in_mst[u] = True
        # Find parent: the node in MST that gave min_dist[u]
        # We need to track this. Let's use a different approach.
        
        min_dist = np.minimum(min_dist, distance_matrix[u])
        min_dist[in_mst] = np.inf
    
    # This approach is getting messy. Let's use a cleaner Prim's with edge tracking.
    in_mst = np.zeros(n, dtype=bool)
    in_mst[0] = True
    # min_edge[i] = (distance, from_node) for node i not in MST
    min_edge_dist = distance_matrix[0].copy()
    min_edge_from = np.zeros(n, dtype=int)  # from node 0 initially
    
    mst_adj = np.zeros((n, n), dtype=bool)
    
    for _ in range(n - 1):
        # Find node with min edge distance
        min_edge_dist_masked = min_edge_dist.copy()
        min_edge_dist_masked[in_mst] = np.inf
        u = np.argmin(min_edge_dist_masked)
        
        in_mst[u] = True
        v = min_edge_from[u]
        mst_adj[u, v] = True
        mst_adj[v, u] = True
        
        # Update min edges for all other nodes
        for w in range(n):
            if not in_mst[w] and distance_matrix[u, w] < min_edge_dist[w]:
                min_edge_dist[w] = distance_matrix[u, w]
                min_edge_from[w] = u
    
    # Set heuristic: high for MST edges, low for others
    # Larger heuristic = more attractive
    heuristic = np.ones((n, n), dtype=float)
    heuristic[mst_adj] = 10.0
    
    # Ensure diagonal is handled (self-loops don't matter much, but keep consistent)
    np.fill_diagonal(heuristic, 1.0)
    
    # Ensure no non-positive values (though 10.0 and 1.0 are both positive)
    heuristic = np.where(heuristic > 0, heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
