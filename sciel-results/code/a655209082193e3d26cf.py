import numpy as np
import time

# Start time for the entire evaluation run to enforce deadline
_start_time = time.time()

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    # Deadline guard: If we are close to the 120s limit, fallback to a very fast heuristic
    if time.time() - _start_time > 110:
        if len(unvisited_nodes) == 0:
            return depot
        unvisited = np.array(unvisited_nodes)
        feasible_mask = demands[unvisited] <= rest_capacity
        feasible_nodes = unvisited[feasible_mask]
        if len(feasible_nodes) == 0:
            return depot
        # Just pick nearest feasible
        dists = distance_matrix[current_node, feasible_nodes]
        return int(feasible_nodes[np.argmin(dists)])

    if len(unvisited_nodes) == 0:
        return depot

    unvisited = np.array(unvisited_nodes)
    
    # 1. Identify feasible nodes (demand <= remaining capacity)
    feasible_mask = demands[unvisited] <= rest_capacity
    feasible_nodes = unvisited[feasible_mask]
    
    if len(feasible_nodes) == 0:
        return depot
    
    if len(feasible_nodes) == 1:
        return int(feasible_nodes[0])

    best_node = -1
    best_score = float('inf')
    
    # Pre-fetch distance row from current node for speed
    dist_from_curr = distance_matrix[current_node]
    
    # 2. 2-Step Lookahead Heuristic
    # For each feasible node n, estimate the cost of going to n and then to the best next feasible node (or depot)
    for n in feasible_nodes:
        # Cost to reach n
        d_curr_n = dist_from_curr[n]
        
        # Remaining capacity after visiting n
        new_rest = rest_capacity - demands[n]
        
        # Find the best next step from n
        # Nodes that are feasible after visiting n
        mask_next = demands[unvisited] <= new_rest
        
        # We need the minimum distance from n to any node k in unvisited such that:
        # 1. k != n
        # 2. demands[k] <= new_rest
        # If no such k exists, the next step is to the depot.
        
        # Vectorized calculation of distances from n to all unvisited
        d_n_to_unvisited = distance_matrix[n, unvisited].copy()
        
        # Mask out n itself
        d_n_to_unvisited[unvisited == n] = float('inf')
        
        # Mask out nodes that don't fit in the new capacity
        d_n_to_unvisited[~mask_next] = float('inf')
        
        min_next_dist = np.min(d_n_to_unvisited)
        
        if min_next_dist == float('inf'):
            # No other feasible node, must return to depot
            next_cost = distance_matrix[n, depot]
        else:
            next_cost = min_next_dist
            
        total_score = d_curr_n + next_cost
        
        if total_score < best_score:
            best_score = total_score
            best_node = n

    return int(best_node)
