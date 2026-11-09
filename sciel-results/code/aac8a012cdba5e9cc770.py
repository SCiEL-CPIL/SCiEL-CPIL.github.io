"""Capacitated VRP next-node heuristic: savings core + cluster look-ahead."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_MODULE_START = time.time()
TIME_LIMIT = 110.0  # safety margin below the 120s/150s evaluator limit


def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    # Deadline guard: fall back to a fast, safe choice before the limit.
    elapsed = time.time() - _MODULE_START
    if elapsed > TIME_LIMIT:
        if len(unvisited_nodes) == 0:
            return depot
        feasible = [u for u in unvisited_nodes if demands[u] <= rest_capacity]
        if not feasible:
            return depot
        return int(min(feasible, key=lambda u: distance_matrix[current_node, u]))

    if len(unvisited_nodes) == 0:
        return depot

    # Filter by capacity feasibility.
    unvisited_arr = np.asarray(unvisited_nodes)
    feasible_mask = demands[unvisited_arr] <= rest_capacity
    if not np.any(feasible_mask):
        return depot
    feasible_nodes = unvisited_arr[feasible_mask]
    n_feasible = len(feasible_nodes)

    if n_feasible == 1:
        return int(feasible_nodes[0])

    # Distances from current node and from each candidate to the depot.
    dists_from_current = distance_matrix[current_node, feasible_nodes]
    dists_to_depot = distance_matrix[feasible_nodes, depot]

    # Capacity-tightness ratio: 1.0 = plenty of room, 0.0 = almost empty.
    total_demand = float(np.sum(demands[unvisited_arr]))
    if total_demand > 0:
        cap_ratio = rest_capacity / total_demand
    else:
        cap_ratio = 1.0
    cap_ratio = min(1.0, max(0.0, cap_ratio))

    # Look-ahead: for each candidate, distance to its nearest other unvisited
    # node (or the depot). Cheap O(n^2) with n <= 50, and it encodes whether
    # the candidate can chain into a cluster instead of dead-ending.
    other_unvisited = np.setdiff1d(unvisited_arr, feasible_nodes, assume_unique=False)
    look_ahead = np.empty(n_feasible)
    for i in range(n_feasible):
        node = int(feasible_nodes[i])
        if len(other_unvisited) > 0:
            d_to_others = float(np.min(distance_matrix[node, other_unvisited]))
        else:
            d_to_others = 0.0
        look_ahead[i] = min(d_to_others, dists_to_depot[i])

    # Weights:
    #   w_dep : penalty for ending far from depot, grows as capacity tightens.
    #   w_la  : look-ahead weight, modest; encourages tight clusters.
    w_dep = 0.35 * (1.0 - cap_ratio)
    w_la = 0.15

    # Savings core: minimize d(current,j) - d(j,depot)  (maximizes Clarke-Wright savings).
    # Add the adaptive depot penalty and the cluster look-ahead term.
    scores = dists_from_current - dists_to_depot + w_dep * dists_to_depot + w_la * look_ahead

    best_idx = int(np.argmin(scores))
    return int(feasible_nodes[best_idx])
# EVOLVE-BLOCK-END
