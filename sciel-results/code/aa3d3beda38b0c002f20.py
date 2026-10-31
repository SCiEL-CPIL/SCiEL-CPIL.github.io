# EVOLVE-BLOCK-START
import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.sparse import csr_matrix


def _compute_mst_cost(nodes, distance_matrix):
    """
    Compute the total weight of the Minimum Spanning Tree (MST) of the given nodes
    using the provided distance matrix.
    """
    n = len(nodes)
    if n <= 1:
        return 0.0
    
    sub_matrix = distance_matrix[np.ix_(nodes, nodes)]
    
    try:
        mst = minimum_spanning_tree(sub_matrix)
        mst_weight = np.sum(mst.data) if mst.nnz > 0 else 0.0
        return float(mst_weight)
    except Exception:
        return float('inf')


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Selects the next node in a TSP tour using a heuristic that combines
    immediate distance with an MST-based estimate of the residual tour cost.
    The residual estimate uses the MST of the remaining unvisited nodes (excluding candidate)
    plus connection costs to anchor the sub-tour.
    """
    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited == 0:
        return -1
        
    if n_unvisited == 1:
        return int(unvisited_nodes[0])

    best_score = float('inf')
    best_node = -1
    
    # Precompute distances from current and destination to all unvisited nodes
    dist_from_current = distance_matrix[current_node, unvisited_nodes]
    dist_from_dest = distance_matrix[destination_node, unvisited_nodes]
    
    # Precompute distance matrix among unvisited nodes for faster access
    unvisited_dist = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
    
    for i in range(n_unvisited):
        candidate = unvisited_nodes[i]
        
        # Immediate cost: distance from current to candidate
        immediate_dist = dist_from_current[i]
        
        # Residual cost estimate:
        # After visiting candidate, remaining nodes are unvisited_nodes without candidate.
        remaining_indices = np.concatenate([np.arange(i), np.arange(i+1, n_unvisited)])
        remaining_nodes = unvisited_nodes[remaining_indices]
        
        if len(remaining_nodes) == 0:
            residual_est = dist_from_dest[i]
        else:
            # Compute MST of remaining nodes
            remaining_submatrix = unvisited_dist[np.ix_(remaining_indices, remaining_indices)]
            try:
                mst = minimum_spanning_tree(remaining_submatrix)
                mst_weight = float(np.sum(mst.data)) if mst.nnz > 0 else 0.0
            except Exception:
                mst_weight = float('inf')
            
            # Add connection costs:
            # 1. Distance from candidate to nearest remaining node
            # 2. Distance from destination to nearest remaining node
            min_dist_cand_rem = np.min(unvisited_dist[i, remaining_indices])
            min_dist_dest_rem = np.min(dist_from_dest[remaining_indices])
            
            residual_est = mst_weight + min_dist_cand_rem + min_dist_dest_rem
        
        total_score = immediate_dist + residual_est
        
        if total_score < best_score:
            best_score = total_score
            best_node = candidate
            
    return int(best_node)
# EVOLVE-BLOCK-END
