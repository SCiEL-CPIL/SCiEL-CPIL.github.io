import numpy as np

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    if len(unvisited_nodes) == 0:
        return depot
    
    # Filter to feasible nodes (demand fits in remaining capacity)
    feasible = []
    for j in unvisited_nodes:
        if demands[j] <= rest_capacity:
            feasible.append(j)
    
    if len(feasible) == 0:
        return depot
    
    feasible = np.array(feasible)
    
    # Heuristic: Minimize distance to next node + distance from next node to depot
    # This balances immediate travel cost with the anticipated return cost,
    # encouraging routes that stay "close" to the depot and don't stray too far.
    
    dist_c_j = distance_matrix[current_node, feasible]
    dist_j_dep = distance_matrix[feasible, depot]
    
    scores = dist_c_j + dist_j_dep
    
    # Pick the minimum score
    best_idx = np.argmin(scores)
    return int(feasible[best_idx])
