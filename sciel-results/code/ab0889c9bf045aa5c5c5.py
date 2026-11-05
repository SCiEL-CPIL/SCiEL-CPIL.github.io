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

    # Feasible: demand fits in remaining capacity
    unvisited_arr = np.asarray(unvisited_nodes, dtype=np.intp)
    feas_mask = demands[unvisited_arr] <= rest_capacity
    if not np.any(feas_mask):
        return int(depot)

    feas = unvisited_arr[feas_mask]
    k = len(feas)

    # Distance from current node to each feasible candidate
    d_cur = distance_matrix[current_node, feas]

    if k == 1:
        return int(feas[0])

    # Clarke-Wright savings heuristic:
    # savings[i] = d(current, depot) + d(depot, feas[i]) - d(current, feas[i])
    # Higher savings means visiting this node saves more distance compared to
    # going back to depot and then visiting it separately.
    d_cur_depot = distance_matrix[current_node, depot]
    d_depot_feas = distance_matrix[depot, feas]
    savings = d_cur_depot + d_depot_feas - d_cur

    # Also consider capacity: prefer nodes that leave more room for future visits
    # Normalize demands to [0,1] range for blending
    d_feas = demands[feas]
    max_d = d_feas.max()
    if max_d > 0:
        norm_d = d_feas / max_d
    else:
        norm_d = np.zeros_like(d_feas, dtype=float)

    # Combine: minimize (d_cur - beta * savings + gamma * norm_d)
    # Or equivalently: maximize (savings - (1/beta)*d_cur - (gamma/beta)*norm_d)
    # Let's use: score = d_cur - 0.8 * savings + 0.15 * norm_d * d_cur_depot
    # The savings term rewards compactness, the demand term slightly penalizes
    # high-demand nodes when capacity is tight
    beta = 0.8
    gamma = 0.15
    scores = d_cur - beta * savings + gamma * norm_d * d_cur_depot

    idx = int(np.argmin(scores))
    return int(feas[idx])
# EVOLVE-BLOCK-END
