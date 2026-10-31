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
    if n == 0:
        return 0.0
    if n == 1:
        return 0.0
    
    # Extract submatrix for unvisited nodes
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
    """
    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited == 0:
        return -1
        
    if n_unvisited == 1:
        return int(unvisited_nodes[0])

    # Pre-compute MST cost for the full set of unvisited nodes
    full_mst_cost = _compute_mst_cost(unvisited_nodes, distance_matrix)
    
    best_score = float('inf')
    best_node = -1
    
    # For each candidate, estimate the residual tour cost
    for i in range(n_unvisited):
        candidate = int(unvisited_nodes[i])
        
        # Immediate cost: distance from current to candidate
        immediate_dist = distance_matrix[current_node, candidate]
        
        # Residual cost estimate:
        # After visiting candidate, the remaining unvisited nodes are unvisited_nodes without candidate.
        # We estimate the cost to finish the tour from candidate to destination, visiting all remaining nodes.
        
        # The residual tour cost consists of:
        # 1. The MST of the remaining nodes (unvisited_nodes \ {candidate})
        # 2. Connection from candidate to the remaining cluster
        # 3. Connection from destination to the remaining cluster
        
        remaining_indices = np.array([j for j in range(n_unvisited) if j != i])
        remaining_nodes = unvisited_nodes[remaining_indices]
        
        if len(remaining_nodes) == 0:
            # No remaining nodes, just go to destination
            residual_est = distance_matrix[candidate, destination_node]
        else:
            # Compute MST of remaining nodes
            remaining_mst_cost = _compute_mst_cost(remaining_nodes, distance_matrix)
            
            # Connection costs: candidate to nearest remaining node, destination to nearest remaining node
            min_dist_cand_rem = np.min(distance_matrix[candidate, remaining_nodes])
            min_dist_dest_rem = np.min(distance_matrix[destination_node, remaining_nodes])
            
            # Residual estimate: MST of remaining + connection costs
            residual_est = remaining_mst_cost + min_dist_cand_rem + min_dist_dest_rem
        
        total_score = immediate_dist + residual_est
        
        if total_score < best_score:
            best_score = total_score
            best_node = candidate
            
    return int(best_node)
# EVOLVE-BLOCK-END
