import numpy as np

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited == 0:
        return -1 

    if n_unvisited == 1:
        return int(unvisited_nodes[0])

    # Return-Aware Greedy
    # Cost to go to candidate + Cost to return from candidate to destination
    cost_to_candidate = distance_matrix[current_node, unvisited_nodes]
    cost_to_return = distance_matrix[unvisited_nodes, destination_node]
    
    total_costs = cost_to_candidate + cost_to_return
    
    best_local_idx = np.argmin(total_costs)
    
    return int(unvisited_nodes[best_local_idx])
