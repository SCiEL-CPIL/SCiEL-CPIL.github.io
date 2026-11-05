"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_T0 = time.time()
_DEADLINE = _T0 + 145.0  # safety margin before 150s kill


def _time_ok():
    return (time.time() - _T0) < _DEADLINE


def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    n = len(unvisited_nodes)
    if n == 0:
        return int(depot)

    # Time check
    if not _time_ok():
        # Fallback to nearest feasible or depot if time is up
        # But for 50 nodes, this is unlikely.
        # Just return nearest feasible to be safe and fast.
        unvisited_arr = np.asarray(unvisited_nodes, dtype=np.intp)
        feas_mask = demands[unvisited_arr] <= rest_capacity
        if not np.any(feas_mask):
            return int(depot)
        feas = unvisited_arr[feas_mask]
        if len(feas) == 0:
            return int(depot)
        return int(feas[np.argmin(distance_matrix[current_node, feas])])

    unvisited_arr = np.asarray(unvisited_nodes, dtype=np.intp)
    
    # Feasibility: demand must fit in remaining capacity
    feas_mask = demands[unvisited_arr] <= rest_capacity
    if not np.any(feas_mask):
        return int(depot)

    feas = unvisited_arr[feas_mask]
    m = len(feas)
    
    if m == 1:
        return int(feas[0])

    # Calculate scores based on Clarke-Wright Savings
    # Minimize d(current, j) - d(depot, j)
    d_current = distance_matrix[current_node, feas]
    d_depot = distance_matrix[depot, feas]
    
    scores = d_current - d_depot
    
    idx = np.argmin(scores)
    return int(feas[idx])
# EVOLVE-BLOCK-END
