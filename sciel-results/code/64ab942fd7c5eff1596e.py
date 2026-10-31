# EVOLVE-BLOCK-START
import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.sparse import csr_matrix


def _compute_mst_min_edge(nodes, distance_matrix):
    """
    Compute the minimum edge weight in the Minimum Spanning Tree (MST) of the given nodes.
    This serves as a local density/connectivity estimate for the residual tour cost.
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
        if mst.nnz == 0:
            return 0.0
        # Get the weights of the MST edges
        weights = mst.data
        # Return the minimum edge weight in the MST
        return float(np.min(weights))
    except Exception:
        # Fallback: return a large value
        return float('inf')


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Selects the next node in a TSP tour using a heuristic that combines
    immediate distance with an MST-based residual estimate using the minimum
    edge weight in the MST of remaining unvisited nodes.
    """
    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited == 0:
        return -1
        
    if n_unvisited == 1:
        return int(unvisited_nodes[0])

    best_score = float('inf')
    best_node = -1
    
    for i in range(n_unvisited):
        candidate = unvisited_nodes[i]
        
        # Immediate cost: distance from current to candidate
        immediate_dist = distance_matrix[current_node, candidate]
        
        # Residual cost estimate:
        # After visiting candidate, the remaining unvisited nodes are unvisited_nodes without candidate.
        remaining_indices = np.array([j for j in range(n_unvisited) if j != i])
        remaining_nodes = unvisited_nodes[remaining_indices]
        
        if len(remaining_nodes) == 0:
            # No remaining nodes, just go to destination
            residual_est = distance_matrix[candidate, destination_node]
        else:
            # Use the minimum edge weight in the MST of remaining nodes as a proxy
            # for the "density" or local connectivity cost of the remaining sub-tour.
            min_mst_edge = _compute_mst_min_edge(remaining_nodes, distance_matrix)
            
            # Scale the min edge by the number of remaining nodes to estimate total residual cost
            # The idea: if the MST has small minimum edges, the nodes are well-clustered,
            # suggesting lower residual tour cost.
            n_rem = len(remaining_nodes)
            residual_est = min_mst_edge * n_rem + distance_matrix[candidate, destination_node]
            
            # Also add a term for connecting candidate to the remaining cluster
            min_dist_cand_rem = np.min(distance_matrix[candidate, remaining_nodes])
            residual_est += min_dist_cand_rem
        
        total_score = immediate_dist + residual_est
        
        if total_score < best_score:
            best_score = total_score
            best_node = candidate
            
    return int(best_node)
# EVOLVE-BLOCK-END
