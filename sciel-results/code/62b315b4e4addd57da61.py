# EVOLVE-BLOCK-START
import numpy as np
import time

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select the next node to visit in a TSP tour using a 2-step lookahead heuristic.
    """
    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited == 0:
        return int(current_node)
    
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    
    if n_unvisited == 2:
        n1, n2 = unvisited_nodes
        # Cost if we go to n1 first: current -> n1 -> n2 -> destination
        cost1 = distance_matrix[current_node, n1] + distance_matrix[n1, n2] + distance_matrix[n2, destination_node]
        # Cost if we go to n2 first: current -> n2 -> n1 -> destination
        cost2 = distance_matrix[current_node, n2] + distance_matrix[n2, n1] + distance_matrix[n1, destination_node]
        return int(n1 if cost1 <= cost2 else n2)
    
    # For n_unvisited > 2, use 2-step lookahead
    # For each candidate i in unvisited_nodes:
    # score_i = distance_matrix[current_node, i] + min_{j in unvisited_nodes, j != i} distance_matrix[i, j]
    
    # Vectorized approach:
    # 1. Get distances from current to all unvisited
    dist_from_current = distance_matrix[current_node, unvisited_nodes]
    
    # 2. Get submatrix of distances among unvisited nodes
    sub_dist = distance_matrix[unvisited_nodes][:, unvisited_nodes]
    
    # 3. For each row i in sub_dist, find the minimum distance to any other node j (j != i)
    # Set diagonal to inf to exclude self
    np.fill_diagonal(sub_dist, np.inf)
    min_dist_to_next = np.min(sub_dist, axis=1)
    
    # 4. Combine scores
    scores = dist_from_current + min_dist_to_next
    
    # 5. Pick the candidate with the minimum score
    best_idx = np.argmin(scores)
    return int(unvisited_nodes[best_idx])

# EVOLVE-BLOCK-END
