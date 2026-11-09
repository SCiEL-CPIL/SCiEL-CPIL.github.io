import numpy as np

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    if not unvisited_nodes:
        return depot
    
    # Identify feasible nodes
    feasible_nodes = []
    for node in unvisited_nodes:
        if demands[node] <= rest_capacity:
            feasible_nodes.append(node)
            
    if not feasible_nodes:
        return depot
        
    # Vectorized distance calculation
    feasible_arr = np.array(feasible_nodes)
    d_current = distance_matrix[current_node, feasible_arr]
    d_depot = distance_matrix[depot, feasible_arr]
    
    # Savings Heuristic Score
    # We want to maximize savings: d(current, depot) + d(depot, node) - d(current, node)
    # Since d(current, depot) is constant for this step, we minimize:
    # d(current, node) - d(depot, node)
    scores = d_current - d_depot
    
    min_idx = np.argmin(scores)
    return int(feasible_arr[min_idx])
