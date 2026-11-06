"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global state for deadline tracking and caching
_start_time = None
_deadline = 145.0  # 5s buffer before 150s limit

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    global _start_time
    
    # Initialize deadline tracking on first call
    if _start_time is None:
        _start_time = time.time()
    
    # Safety margin: stop if we've exceeded deadline
    elapsed = time.time() - _start_time
    if elapsed > _deadline:
        return depot
    
    if len(unvisited_nodes) == 0:
        return depot
    
    # Check feasible nodes (demand <= rest_capacity)
    unvisited_arr = np.array(unvisited_nodes)
    feasible_mask = demands[unvisited_arr] <= rest_capacity
    feasible_nodes = unvisited_arr[feasible_mask]
    
    if len(feasible_nodes) == 0:
        return depot
    
    # Clarke-Wright savings heuristic adapted for sequential route construction
    # Savings for linking current_node to candidate c:
    # s = d(depot, current) + d(depot, c) - d(current, c)
    # We want to maximize savings
    
    d_depot_current = distance_matrix[depot, current_node]
    d_depot_cand = distance_matrix[depot, feasible_nodes]
    d_current_cand = distance_matrix[current_node, feasible_nodes]
    
    savings = d_depot_current + d_depot_cand - d_current_cand
    
    # Pick the node with max savings
    idx = np.argmax(savings)
    best_node = feasible_nodes[idx]
    
    return int(best_node)
# EVOLVE-BLOCK-END
