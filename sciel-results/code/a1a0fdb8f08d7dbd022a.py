"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(distance_matrix):
    """
    Compute an edge-attractiveness matrix for ant colony optimization of TSP.
    
    Strategy: Use a Minimum Spanning Tree (MST) based heuristic.
    1. Compute the MST of the complete graph defined by the distance matrix.
    2. Assign high attractiveness to edges in the MST, scaled by inverse distance.
    3. Assign lower attractiveness to non-MST edges, also scaled by inverse distance.
    This guides the ants to construct tours that are topologically similar to the MST,
    which is a strong lower bound and structural guide for the TSP tour.
    """
    n = distance_matrix.shape[0]
    
    # 1. Compute MST using Prim's algorithm
    # dist_matrix is the input
    # visited: boolean array
    # min_dist: array of min distance from visited set to unvisited nodes
    # parent: array to track which node connects the new node to the tree
    
    visited = np.zeros(n, dtype=bool)
    min_dist = np.inf * np.ones(n)
    parent = np.zeros(n, dtype=int)
    
    # Start from node 0
    min_dist[0] = 0.0
    visited[0] = True
    parent[0] = -1
    
    mst_edges = np.zeros((n, n), dtype=bool)
    
    for _ in range(n - 1):
        # Find unvisited node with minimum distance
        # Mask visited nodes with inf
        temp_dist = min_dist.copy()
        temp_dist[visited] = np.inf
        u = np.argmin(temp_dist)
        
        if min_dist[u] == np.inf:
            break # Graph disconnected (should not happen for TSP distance matrix)
            
        visited[u] = True
        
        # Add edge (parent[u], u) to MST
        if parent[u] != -1:
            mst_edges[u, parent[u]] = True
            mst_edges[parent[u], u] = True
            
        # Update min_dist for neighbors of u
        # For all unvisited nodes v, if dist[u, v] < min_dist[v], update
        for v in range(n):
            if not visited[v] and distance_matrix[u, v] < min_dist[v]:
                min_dist[v] = distance_matrix[u, v]
                parent[v] = u
                
    # 2. Construct Heuristic
    # Base heuristic: Inverse distance squared (strong preference for short edges)
    # We use squared inverse to make the heuristic more discriminative for very short edges
    base = 1.0 / (distance_matrix + 1e-9) ** 2
    
    # Boost MST edges
    # Let's multiply MST edges by a factor, say 5.0
    # And keep non-MST edges as base
    heuristic = base.copy()
    heuristic[mst_edges] *= 5.0
    
    # Zero out diagonal
    np.fill_diagonal(heuristic, 0.0)
    
    # Ensure positive values
    heuristic = np.maximum(heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
