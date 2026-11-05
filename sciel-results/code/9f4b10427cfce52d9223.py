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
    
    # Clarke-Wright Savings Heuristic
    # Saving = d(c, depot) + d(depot, j) - d(c, j)
    # We want to maximize Saving.
    # Equivalent to minimizing: d(c, j) - d(depot, j)
    # d(c, depot) is constant for all j in this step, so it doesn't affect argmin.
    
    dist_c_j = distance_matrix[current_node, feasible]
    dist_dep_j = distance_matrix[depot, feasible]
    
    scores = dist_c_j - dist_dep_j
    
    # Pick the minimum score (max savings)
    best_idx = np.argmin(scores)
    return int(feasible[best_idx])
