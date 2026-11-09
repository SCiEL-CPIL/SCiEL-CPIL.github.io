import numpy as np
import scipy.sparse.csgraph as csgraph

def heuristics(distance_matrix):
    n = distance_matrix.shape[0]
    eps = 1e-9
    
    # Base heuristic: inverse distance
    # Ensure no division by zero by adding eps
    base_heuristic = 1.0 / (distance_matrix + eps)
    
    # Set diagonal to 0 or small value? 
    # The problem says "Heuristic entries that are not positive become 1e-9".
    # If we leave diagonal as 1/eps (huge), ants might pick self-loops?
    # "picks the next unvisited node". So self-loops are never picked.
    # But to be safe, let's set diagonal to 0 or 1e-9.
    np.fill_diagonal(base_heuristic, 1e-9)
    
    # Compute Minimum Spanning Tree
    # distance_matrix is a square array.
    # scipy.sparse.csgraph.minimum_spanning_tree expects a square matrix or sparse graph.
    # It returns a sparse matrix with 1s where edges are in the MST.
    # We need to ensure the graph is connected. Random points in unit square are connected.
    
    # Note: minimum_spanning_tree uses the values as weights.
    # We want the MST of the graph where edge weight is distance.
    # So we pass distance_matrix directly.
    
    # However, distance_matrix has 0 on diagonal. This is fine.
    # But wait, if there are duplicate points, distance is 0. MST might pick 0-weight edges.
    # That's fine.
    
    try:
        mst_sparse = csgraph.minimum_spanning_tree(distance_matrix)
        mst_mask = mst_sparse.toarray() > 0
    except Exception:
        # Fallback if MST fails
        mst_mask = np.zeros((n, n), dtype=bool)
        
    # Boost MST edges
    boost_factor = 10.0
    heuristic = base_heuristic.copy()
    heuristic[mst_mask] *= boost_factor
    
    # Optional: Also boost very short edges?
    # The MST already captures the shortest structural connections.
    # Let's stick to MST boost.
    
    return heuristic
