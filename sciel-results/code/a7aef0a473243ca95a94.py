"""Rewritten program: Savings-based (Clarke-Wright) heuristic for node selection."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Time guard: track when the process started
_START_TIME = time.time()
_TIME_LIMIT = 140.0  # Safety margin before 150s kill


def _time_exceeded():
    return (time.time() - _START_TIME) > _TIME_LIMIT


def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Pick the next customer (or depot 0 to close) using a savings-based heuristic.
    
    Savings for appending j after i: dist(depot,i) + dist(depot,j) - dist(i,j).
    We prefer feasible customers with high savings (i.e., low marginal cost).
    Tie-break: prefer nearer customers.
    """
    if _time_exceeded():
        # Fallback: just return nearest feasible to be safe
        unvisited = np.asarray(unvisited_nodes, dtype=int)
        if len(unvisited) == 0:
            return 0
        feasible = unvisited[demands[unvisited] <= rest_capacity]
        if len(feasible) == 0:
            return 0
        return int(feasible[np.argmin(distance_matrix[current_node, feasible])])

    unvisited = np.asarray(unvisited_nodes, dtype=int)
    n_unvisited = len(unvisited)

    if n_unvisited == 0:
        return 0

    # Feasibility: demand must fit in remaining capacity
    feasible_mask = demands[unvisited] <= rest_capacity
    feasible = unvisited[feasible_mask]

    if len(feasible) == 0:
        return 0

    # If only one feasible node, take it
    if len(feasible) == 1:
        return int(feasible[0])

    # Compute savings for each feasible node j:
    # savings[j] = dist(depot, current_node) + dist(depot, j) - dist(current_node, j)
    # Higher savings = better (less marginal cost)
    dist_depot_current = distance_matrix[depot, current_node]
    dist_depot_feasible = distance_matrix[depot, feasible]
    dist_current_feasible = distance_matrix[current_node, feasible]

    savings = dist_depot_current + dist_depot_feasible - dist_current_feasible

    # We want to MAXIMIZE savings (minimize marginal cost)
    # Use a composite score: primarily savings, with a small tie-break toward nearer nodes
    # to avoid pathological ties
    # Normalize distances to be in a reasonable range for tie-breaking
    max_dist = np.max(dist_current_feasible) if np.max(dist_current_feasible) > 0 else 1.0
    tie_break = -dist_current_feasible / max_dist * 0.01  # small penalty for far nodes

    score = savings + tie_break

    best_idx = int(np.argmax(score))
    return int(feasible[best_idx])
# EVOLVE-BLOCK-END
