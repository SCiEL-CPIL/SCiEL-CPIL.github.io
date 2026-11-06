"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = None


def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    global _start_time
    
    if _start_time is None:
        _start_time = time.time()
    
    # Safety margin: 120s limit, stop at 118s
    if time.time() - _start_time > 118:
        return int(depot)
    
    if len(unvisited_nodes) == 0:
        return int(depot)
    
    unvisited_arr = np.array(unvisited_nodes, dtype=int)
    
    # Filter feasible nodes by capacity
    feasible_mask = demands[unvisited_arr] <= rest_capacity
    feasible_nodes = unvisited_arr[feasible_mask]
    
    if len(feasible_nodes) == 0:
        return int(depot)
    
    # If only one feasible node, take it
    if len(feasible_nodes) == 1:
        return int(feasible_nodes[0])
    
    # Heuristic: Combine immediate distance, estimated future cost, and a dynamic depot-pull term.
    # The depot-pull term increases as remaining capacity decreases, encouraging the route
    # to close sooner when the vehicle is nearly full, which can improve packing efficiency
    # and reduce the total number of routes or deadhead distance.
    
    dist_from_current = distance_matrix[current_node, feasible_nodes]
    dist_to_depot = distance_matrix[feasible_nodes, depot]
    demands_feasible = demands[feasible_nodes]
    
    n = len(feasible_nodes)
    scores = np.zeros(n)
    
    # Dynamic alpha: encourages return to depot when capacity is low.
    # Scale: assume max capacity ~50. If rest_cap < 25, start pulling.
    alpha = max(0.0, 1.0 - rest_capacity / 25.0)
    
    for idx in range(n):
        rem_cap = rest_capacity - demands_feasible[idx]
        
        # Estimate future cost
        if rem_cap <= 0:
            future_cost = dist_to_depot[idx]
        else:
            # Check if any other node fits in remaining capacity
            other_mask = demands_feasible <= rem_cap
            other_mask[idx] = False  # exclude self
            if np.any(other_mask):
                other_nodes = feasible_nodes[other_mask]
                future_cost = np.min(distance_matrix[feasible_nodes[idx], other_nodes])
            else:
                future_cost = dist_to_depot[idx]
        
        # Score: immediate cost + future cost + depot pull
        scores[idx] = dist_from_current[idx] + future_cost + alpha * dist_to_depot[idx]
        
    best_idx = np.argmin(scores)
    return int(feasible_nodes[best_idx])
# EVOLVE-BLOCK-END
