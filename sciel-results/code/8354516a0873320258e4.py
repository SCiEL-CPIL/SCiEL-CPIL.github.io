"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_TIME_LIMIT = 145.0  # Safety margin for 150s limit

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    # Deadline guard
    if time.time() - _START_TIME > _TIME_LIMIT:
        return depot

    if len(unvisited_nodes) == 0:
        return depot
    
    # Filter to feasible nodes (demand fits in remaining capacity)
    # Using list comprehension for speed, then numpy for vectorization
    feasible_nodes = []
    for j in unvisited_nodes:
        if demands[j] <= rest_capacity:
            feasible_nodes.append(j)
            
    if len(feasible_nodes) == 0:
        return depot
        
    feasible_arr = np.array(feasible_nodes)
    
    # Calculate distances
    # dist_curr: distance from current node to each feasible node
    dist_curr = distance_matrix[current_node, feasible_arr]
    # dist_depot: distance from each feasible node to depot
    dist_depot = distance_matrix[feasible_arr, depot]
    
    # Heuristic: Lookahead Nearest Neighbor
    # Minimize the cost of moving to j and then returning to depot
    # This avoids "dead ends" far from the depot
    scores = dist_curr + dist_depot
    
    # Find the index of the minimum score
    min_idx = np.argmin(scores)
    
    return int(feasible_arr[min_idx])
# EVOLVE-BLOCK-END
