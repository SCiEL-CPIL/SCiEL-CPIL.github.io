"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_T0 = time.time()
_DEADLINE = _T0 + 145.0

def _time_ok():
    return (time.time() - _T0) < _DEADLINE

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    # If time is running out, return depot to close route safely
    if not _time_ok():
        return int(depot)
    
    if len(unvisited_nodes) == 0:
        return int(depot)
    
    # Convert to numpy arrays for efficiency
    unvisited_arr = np.asarray(unvisited_nodes, dtype=np.intp)
    
    # Filter feasible nodes based on capacity
    feas_mask = demands[unvisited_arr] <= rest_capacity
    if not np.any(feas_mask):
        return int(depot)
    
    feas_nodes = unvisited_arr[feas_mask]
    
    # Calculate distances
    # d[current, depot] is constant for this step
    d_cur_dep = distance_matrix[current_node, depot]
    
    # d[depot, feas_nodes]
    d_dep_feas = distance_matrix[depot, feas_nodes]
    
    # d[current, feas_nodes]
    d_cur_feas = distance_matrix[current_node, feas_nodes]
    
    # Savings heuristic:
    # Saving = d[current, depot] + d[depot, c] - d[current, c]
    # We want to maximize this.
    savings = d_cur_dep + d_dep_feas - d_cur_feas
    
    # Select node with maximum saving
    # In case of ties, np.argmax picks the first one.
    # We could add a small tie-breaker preferring smaller d_cur_feas if savings are equal,
    # but floating point exact ties are rare.
    best_idx = np.argmax(savings)
    best_node = feas_nodes[best_idx]
    
    return int(best_node)
# EVOLVE-BLOCK-END
