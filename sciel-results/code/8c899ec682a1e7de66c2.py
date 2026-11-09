import numpy as np
import time

_start_time = time.time()
_DEADLINE = 145.0

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    if time.time() > _DEADLINE:
        return 0

    if not unvisited_nodes:
        return 0

    # Filter feasible nodes
    # Assume demands is accessible by index
    feasible_nodes = [n for n in unvisited_nodes if demands[n] <= rest_capacity]
    
    if not feasible_nodes:
        return 0

    # Convert to numpy array for fast indexing
    feasible_arr = np.array(feasible_nodes)
    
    # Distances from current to feasible nodes
    d_c_n = distance_matrix[current_node, feasible_arr]
    
    # Distances from depot to feasible nodes
    d_d_n = distance_matrix[depot, feasible_arr]
    
    # Distance from current to depot (scalar)
    d_c_d = distance_matrix[current_node, depot]
    
    # Savings: d(c, d) + d(d, n) - d(c, n)
    savings = d_c_d + d_d_n - d_c_n
    
    # Score: Savings - Distance(c, n)
    # This balances the benefit of not returning to depot against the cost of travel.
    scores = savings - d_c_n
    
    # Select node with highest score
    best_idx = np.argmax(scores)
    return int(feasible_arr[best_idx])
