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
    3. Score each feasible candidate using a multi-criteria heuristic:
       - Immediate distance from current_node to candidate
       - Lookahead: min distance from candidate to any other feasible node or depot
         (penalizes dead-ends)
       - Capacity utilization: prefer nodes with higher demand when capacity is tight
         (encourages fuller trucks, fewer routes)
    4. If time is running out, fall back to nearest feasible node.
    """
    if _time_exceeded():
        # Fallback: nearest feasible node
        feasible = [n for n in unvisited_nodes if demands[n] <= rest_capacity]
        if not feasible:
            return int(depot)
        if len(feasible) == 1:
            return int(feasible[0])
        feasible_arr = np.array(feasible)
        dists = distance_matrix[current_node, feasible_arr]
        return int(feasible_arr[np.argmin(dists)])

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
    
    # If very few unvisited nodes, just pick nearest
    if n_feas <= 2:
        return int(feasible_arr[np.argmin(immediate)])
    
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
    # 
    # Weights:
    # alpha = weight for lookahead
    # gamma = weight for capacity utilization
    
    alpha = 0.4
    gamma = 0.15
    
    scores = immediate + alpha * lookahead_cost - gamma * demand_ratio
    
    # Pick the candidate with the lowest score
    best_idx = int(np.argmin(scores))
    return int(feasible_arr[best_idx])
# EVOLVE-BLOCK-END
