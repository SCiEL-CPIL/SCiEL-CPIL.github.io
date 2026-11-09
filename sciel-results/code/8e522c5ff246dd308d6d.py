import numpy as np

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    n = len(unvisited_nodes)
    if n == 0:
        return -1 # Should not happen
    if n == 1:
        return int(unvisited_nodes[0])
    
    # Get distances from current to all unvisited
    d_current = distance_matrix[current_node, unvisited_nodes]
    
    if n == 2:
        # Exact calculation for 2 nodes
        # unvisited_nodes[0] and unvisited_nodes[1]
        # We need to check both orders
        # Order 1: current -> 0 -> 1 -> dest
        # Order 2: current -> 1 -> 0 -> dest
        j0 = unvisited_nodes[0]
        j1 = unvisited_nodes[1]
        
        cost0 = d_current[0] + distance_matrix[j0, j1] + distance_matrix[j1, destination_node]
        cost1 = d_current[1] + distance_matrix[j1, j0] + distance_matrix[j0, destination_node]
        
        if cost0 < cost1:
            return int(j0)
        else:
            return int(j1)
            
    # n > 2
    # For each candidate j, calculate score = d_current[j] + min_{k != j} (d_jk + d_k_dest)
    
    # Vectorized approach for speed?
    # d_matrix_unvisited = distance_matrix[unvisited_nodes, :][:, unvisited_nodes]
    # d_to_dest = distance_matrix[unvisited_nodes, destination_node]
    
    # We need min over k != j of (d_jk + d_k_dest)
    # This is a row-wise minimum of (D + d_to_dest[:, None]) with diagonal masked.
    
    D = distance_matrix[unvisited_nodes, unvisited_nodes]
    d_dest = distance_matrix[unvisited_nodes, destination_node]
    
    # Add d_dest to each row
    # Score_matrix[i, k] = D[i, k] + d_dest[k]
    Score_matrix = D + d_dest[np.newaxis, :]
    
    # Mask diagonal (k == i) with inf
    np.fill_diagonal(Score_matrix, np.inf)
    
    # Min over columns (axis=1) gives min cost for each j
    min_next_costs = np.min(Score_matrix, axis=1)
    
    # Total score for j is d_current[j] + min_next_costs[j]
    total_scores = d_current + min_next_costs
    
    best_idx = np.argmin(total_scores)
    return int(unvisited_nodes[best_idx])
