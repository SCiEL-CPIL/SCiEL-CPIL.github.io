"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Selects the next node using a 2-step lookahead heuristic.
    Minimizes the sum of the distance to the current node and the minimum 
    distance from the candidate to any other unvisited node (or destination 
    if it is the last unvisited node).
    """
    unvisited_nodes = np.asarray(unvisited_nodes)
    k = len(unvisited_nodes)
    
    if k == 0:
        # Should not happen if logic is correct, but safe fallback
        return destination_node
        
    if k == 1:
        # Only one node left, must go there, then to destination
        # The score is effectively fixed for the choice, so just return it
        return int(unvisited_nodes[0])

    # Get distances from current_node to all unvisited nodes
    entry_costs = distance_matrix[current_node, unvisited_nodes]
    
    # Get the submatrix of distances between unvisited nodes
    # D[i, j] is distance between unvisited_nodes[i] and unvisited_nodes[j]
    D = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
    
    # To find the minimum distance from each node to any OTHER unvisited node:
    # 1. Set diagonal to infinity
    np.fill_diagonal(D, np.inf)
    # 2. Take min along axis 1 (for each row/node)
    exit_costs = np.min(D, axis=1)
    
    # Total score: entry cost + estimated exit cost
    total_costs = entry_costs + exit_costs
    
    # Find the index of the minimum total cost
    best_idx_in_unvisited = np.argmin(total_costs)
    
    return int(unvisited_nodes[best_idx_in_unvisited])
# EVOLVE-BLOCK-END
