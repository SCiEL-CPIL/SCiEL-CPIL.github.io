"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Deadline guard
_START_TIME = time.time()
_DEADLINE = 150.0
_SAFETY_MARGIN = 3.0


def _time_exceeded():
    return (time.time() - _START_TIME) > (_DEADLINE - _SAFETY_MARGIN)


def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Heuristic for capacitated vehicle routing:
    Pick the next customer to visit, or return to depot (0) to close the route.
    
    Strategy:
    1. Filter unvisited nodes that fit within rest_capacity.
    2. If none fit, return depot.
    3. If current_node is depot, must pick a customer (unless no feasible).
    4. Score each feasible candidate using a multi-criteria heuristic.
    5. Compare best candidate score vs depot return score.
    """
    if len(unvisited_nodes) == 0:
        return int(depot)

    # Filter feasible nodes
    feasible = [n for n in unvisited_nodes if demands[n] <= rest_capacity]
    
    if not feasible:
        return int(depot)
    
    if len(feasible) == 1:
        return int(feasible[0])
    
    feasible_arr = np.array(feasible)
    n_feas = len(feasible_arr)
    
    # Immediate distance from current node to each feasible candidate
    immediate = distance_matrix[current_node, feasible_arr]
    
    # If we are at the depot, we cannot return to the depot. We must pick a customer.
    if current_node == depot:
        # Pick the nearest feasible node to start the route
        best_idx = int(np.argmin(immediate))
        return int(feasible_arr[best_idx])

    if _time_exceeded():
        # Fallback: nearest feasible node
        best_idx = int(np.argmin(immediate))
        return int(feasible_arr[best_idx])

    # Lookahead: for each candidate, find the minimum distance to any other
    # feasible unvisited node or the depot
    # This penalizes candidates that are "dead-ends" far from everything else
    
    # Vectorized: get submatrix of distances between feasible nodes
    feas_dist = distance_matrix[np.ix_(feasible_arr, feasible_arr)]  # (n, n)
    dist_to_depot = distance_matrix[feasible_arr, depot]  # (n,)
    
    # For each candidate i, min distance to other feasible nodes (excluding self)
    np.fill_diagonal(feas_dist, np.inf)
    min_to_other_feasible = np.min(feas_dist, axis=1)  # (n,)
    
    # Lookahead cost: min of (min distance to other feasible nodes, distance to depot)
    lookahead_cost = np.minimum(min_to_other_feasible, dist_to_depot)
    
    # Capacity utilization: prefer nodes with higher demand to fill trucks
    # This reduces the number of routes needed
    demand_ratio = demands[feasible_arr] / max(rest_capacity, 1)
    
    # Composite score:
    # - Primary: immediate distance (minimize)
    # - Secondary: lookahead cost (minimize, avoid dead-ends)
    # - Tertiary: capacity utilization (maximize, so subtract)
    
    alpha = 0.55
    gamma = 0.12
    
    scores = immediate + alpha * lookahead_cost - gamma * demand_ratio
    
    # Also consider returning to depot: compare depot "score" vs best candidate
    dist_to_depot_from_current = distance_matrix[current_node, depot]
    
    # Estimate: if we return now, how much capacity is "wasted"?
    if len(unvisited_nodes) > 0:
        avg_remaining_demand = float(np.mean(demands[unvisited_nodes]))
        capacity_pressure = rest_capacity / max(avg_remaining_demand, 1.0)
        depot_score = dist_to_depot_from_current + 0.3 * max(0.0, 1.0 - capacity_pressure) * 10.0
    else:
        depot_score = dist_to_depot_from_current
    
    best_score = float(np.min(scores))
    
    if depot_score < best_score:
        return int(depot)
    
    # Pick the candidate with the lowest score
    best_idx = int(np.argmin(scores))
    return int(feasible_arr[best_idx])
# EVOLVE-BLOCK-END
