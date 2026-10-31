"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select the next node using a 2-step lookahead heuristic.
    
    For each candidate node u in unvisited_nodes, we estimate the cost of moving from
    current_node to u, and then from u to the best possible next node v.
    
    If u is the last node, the cost includes the return to destination.
    If u is the second-to-last node, the cost includes the move to the last node and the return to destination.
    Otherwise, the cost is dist(current, u) + dist(u, nearest_v_in_remaining).
    
    This approach avoids local minima where a node is close to the current node but
    leads to a very expensive subsequent move.
    """
    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited == 0:
        return int(destination_node)
    
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    
    # Precompute distances from current to all unvisited
    dist_current_to_unvisited = distance_matrix[current_node, unvisited_nodes]
    
    # We will evaluate each candidate u
    best_score = np.inf
    best_node = unvisited_nodes[0]
    
    # Create a mask or list to handle removal of u from unvisited for the lookahead
    # Since n is small (<=50), we can loop and create sub-arrays
    
    for i, u in enumerate(unvisited_nodes):
        # Cost of first step
        cost_1 = dist_current_to_unvisited[i]
        
        # Remaining nodes after visiting u
        # We need to find the best next node v from the remaining unvisited nodes
        if n_unvisited == 2:
            # u is one of two. The other one is the only choice for v.
            # The other index is 1-i
            v = unvisited_nodes[1 - i]
            # Cost: current -> u -> v -> destination
            cost_2 = distance_matrix[u, v]
            cost_3 = distance_matrix[v, destination_node]
            total_cost = cost_1 + cost_2 + cost_3
        else:
            # n_unvisited > 2
            # Create array of remaining indices
            # We want to find min distance from u to any w in unvisited_nodes excluding u
            # We can use the submatrix or just compute distances from u to all unvisited and ignore u
            dist_u_to_all = distance_matrix[u, unvisited_nodes]
            # Set distance to self to inf
            dist_u_to_all[i] = np.inf
            # The best next node v is the one with min distance
            min_dist_2 = np.min(dist_u_to_all)
            total_cost = cost_1 + min_dist_2
            
            # Optional: If we want to be more accurate for the last few steps, we could check
            # if the best v leads to a bad end, but 2-step is a standard distinct heuristic.
            # To improve performance and robustness, we stick to 2-step.
            
        if total_cost < best_score:
            best_score = total_cost
            best_node = u
            
    return int(best_node)
# EVOLVE-BLOCK-END
