"""Precomputed global nearest-neighbor tour with capacity-aware sequential serving."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_TIME_LIMIT = 140.0  # Safety margin before 150s kill

# Cache: keyed by (id of distance_matrix, n) to avoid recomputing tour across calls
_TOUR_CACHE = {}

def _time_remaining():
    return _TIME_LIMIT - (time.time() - _START_TIME)

def _build_nn_tour(distance_matrix, depot, n):
    """
    Build a global nearest-neighbor tour starting from depot.
    Returns a list of customer nodes in the order they are visited (excluding depot).
    This is computed once per distance_matrix and cached.
    """
    key = (id(distance_matrix), n)
    if key in _TOUR_CACHE:
        return _TOUR_CACHE[key]

    # Build tour via nearest neighbor
    visited = [False] * n
    visited[depot] = True
    tour = []
    current = depot
    for _ in range(n - 1):
        best_next = -1
        best_dist = float('inf')
        for j in range(n):
            if not visited[j]:
                d = distance_matrix[current, j]
                if d < best_dist:
                    best_dist = d
                    best_next = j
        if best_next == -1:
            break
        visited[best_next] = True
        tour.append(best_next)
        current = best_next

    _TOUR_CACHE[key] = tour
    return tour

def _fast_nearest(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """Fast fallback: nearest feasible neighbor."""
    if len(unvisited_nodes) == 0:
        return depot
    feasible = [n for n in unvisited_nodes if demands[n] <= rest_capacity]
    if not feasible:
        return depot
    if len(feasible) == 1:
        return int(feasible[0])
    d = distance_matrix[current_node, feasible]
    return int(feasible[np.argmin(d)])

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Main entry point.
    Strategy: Precompute a global NN tour once. Serve customers in tour order,
    but only if they are feasible (demand <= rest_capacity). If the next tour
    node is infeasible, skip to the next feasible tour node. If no feasible
    tour node remains, return depot to close the route.
    This creates compact spatial sectors because the NN tour naturally groups
    nearby customers, and capacity splitting keeps routes tight.
    """
    if _time_remaining() < 0.1:
        return _fast_nearest(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix)

    if len(unvisited_nodes) == 0:
        return depot

    n = len(demands)
    unvisited_set = set(unvisited_nodes)

    # Get or build the global tour
    tour = _build_nn_tour(distance_matrix, depot, n)

    # Find the first node in tour order that is both unvisited and feasible
    for node in tour:
        if node in unvisited_set:
            if demands[node] <= rest_capacity:
                return int(node)

    # No feasible node in tour order -- fallback to nearest feasible
    return _fast_nearest(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix)

# EVOLVE-BLOCK-END
