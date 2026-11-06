import numpy as np
import time

_start_time = time.time()
TIME_LIMIT = 115.0

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    if time.time() - _start_time > TIME_LIMIT:
        return depot

    if len(unvisited_nodes) == 0:
        return depot

    # Ensure numpy arrays for fast indexing
    if not isinstance(demands, np.ndarray):
        demands = np.array(demands)
    if not isinstance(distance_matrix, np.ndarray):
        distance_matrix = np.array(distance_matrix)
        
    unvisited = np.array(unvisited_nodes)
    
    # Filter by capacity
    feasible_mask = demands[unvisited] <= rest_capacity
    
    if not np.any(feasible_mask):
        return depot

    feasible_nodes = unvisited[feasible_mask]
    n_feasible = len(feasible_nodes)
    
    if n_feasible == 1:
        return int(feasible_nodes[0])

    # Distance from current to feasible
    d_cur = distance_matrix[current_node, feasible_nodes]
    
    # Distance matrix among feasible nodes
    d_feas = distance_matrix[np.ix_(feasible_nodes, feasible_nodes)]
    
    # Distance from feasible to depot
    d_dep = distance_matrix[feasible_nodes, depot]
    
    # We want to minimize: d(i, j) + min_{k != j} (d(j, k) + d(k, depot))
    
    # Construct cost matrix for the "next step" part
    # M[j, k] = d(j, k) + d(k, depot)
    M = d_feas + d_dep[np.newaxis, :]
    
    # Exclude self-loop k=j
    np.fill_diagonal(M, np.inf)
    
    # Find min next cost for each j
    next_costs = np.min(M, axis=1)
    
    # Total score
    scores = d_cur + next_costs
    
    min_idx = np.argmin(scores)
    
    return int(feasible_nodes[min_idx])
