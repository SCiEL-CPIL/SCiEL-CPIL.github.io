"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = 150.0
_SAFETY_MARGIN = 3.0


def _time_exceeded():
    return (time.time() - _START_TIME) > (_DEADLINE - _SAFETY_MARGIN)


def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Heuristic for capacitated VRP: pick next customer or depot (0) to close route.
    
    Strategy:
      1. Filter unvisited nodes that fit in rest_capacity.
      2. If none fit, return depot.
      3. Score each feasible node using:
         - immediate distance from current_node
         - a savings penalty: how much extra distance is added vs going straight to depot
         - a clustering bonus: proximity to other unvisited feasible nodes
    """
    if _time_exceeded():
        return 0

    unvisited = list(unvisited_nodes)
    if not unvisited:
        return 0

    # Filter by capacity
    feasible = [n for n in unvisited if demands[n] <= rest_capacity]
    if not feasible:
        return 0

    feasible_arr = np.array(feasible, dtype=np.int64)
    n_feasible = len(feasible_arr)
    if n_feasible == 1:
        return int(feasible_arr[0])

    # Immediate distances from current_node
    dist_from_current = distance_matrix[current_node, feasible_arr]

    # Savings penalty: dist(current->node) + dist(node->depot) - dist(current->depot)
    # This is the extra cost of inserting this node before going to depot.
    dist_to_depot = distance_matrix[feasible_arr, depot]
    dist_current_to_depot = distance_matrix[current_node, depot]
    savings_penalty = dist_from_current + dist_to_depot - dist_current_to_depot

    # Clustering: min distance from each feasible node to any OTHER feasible node
    # Build sub-matrix among feasible nodes only
    sub_dist = distance_matrix[np.ix_(feasible_arr, feasible_arr)]
    # Set diagonal to inf to exclude self
    np.fill_diagonal(sub_dist, np.inf)
    cluster_min = np.min(sub_dist, axis=1)

    # Score: weight immediate distance, savings penalty, and clustering
    # Lower score = better
    # We want to minimize: going far now, adding lots of extra distance, and being isolated
    alpha_savings = 0.3
    alpha_cluster = 0.2
    scores = dist_from_current + alpha_savings * savings_penalty + alpha_cluster * cluster_min

    best_idx = np.argmin(scores)
    return int(feasible_arr[best_idx])
# EVOLVE-BLOCK-END
