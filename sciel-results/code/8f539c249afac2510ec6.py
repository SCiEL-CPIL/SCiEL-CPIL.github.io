import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree

def heuristics(distance_matrix):
    n = distance_matrix.shape[0]
    eps = 1e-9
    
    # Compute the Minimum Spanning Tree (MST)
    # The input is a dense symmetric matrix representing a complete graph.
    mst = minimum_spanning_tree(distance_matrix)
    
    # Extract edges from the MST
    mst_coo = mst.tocoo()
    mst_mask = np.zeros((n, n), dtype=bool)
    for i, j in zip(mst_coo.row, mst_coo.col):
        mst_mask[i, j] = True
        mst_mask[j, i] = True
        
    # Base heuristic: Inverse distance squared
    # This emphasizes short edges strongly
    base_heuristic = 1.0 / (distance_matrix + eps) ** 2
    
    # Boost MST edges
    # MST edges are highly likely to be part of the optimal TSP tour
    boost_factor = 4.0
    
    heuristic = base_heuristic.copy()
    heuristic[mst_mask] *= boost_factor
    
    # Ensure all values are positive
    heuristic[heuristic <= 0] = 1e-9
    
    return heuristic
