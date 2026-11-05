# EVOLVE-BLOCK-START
import numpy as np
import time

_T0 = time.time()
_DEADLINE = _T0 + 140.0  # safety margin before 150s kill


def _time_ok():
    return (time.time() - _T0) < _DEADLINE


def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Dynamic regret-minimization heuristic for CVRP.
    
    Score for each candidate = edge_cost - alpha * (edge_cost - mean_edge_cost_of_feasible_set)
    This is equivalent to (1 - alpha) * edge_cost + alpha * mean_edge_cost.
    
    With alpha > 0, we penalize outlier-long edges relative to the average,
    steering the route toward uniform segment lengths and preventing premature
    capacity exhaustion on distant nodes.
    
    We also incorporate a Clarke-Wright-style savings consideration:
    prefer nodes that have high "savings" (depot_to_i + depot_to_j - dist_i_j)
    relative to the current node, which encourages geometrically consistent routes.
    """
    if not _time_ok():
        # Fallback: nearest feasible or depot
        unvisited_arr = np.asarray(unvisited_nodes, dtype=np.intp)
        if len(unvisited_arr) == 0:
            return int(depot)
        feas_mask = demands[unvisited_arr] <= rest_capacity
        if not np.any(feas_mask):
            return int(depot)
        feas = unvisited_arr[feas_mask]
        return int(feas[np.argmin(distance_matrix[current_node, feas])])

    n = len(unvisited_nodes)
    if n == 0:
        return int(depot)

    unvisited_arr = np.asarray(unvisited_nodes, dtype=np.intp)
    k = len(unvisited_arr)

    # Feasibility: demand must fit in remaining capacity
    feas_mask = demands[unvisited_arr] <= rest_capacity
    if not np.any(feas_mask):
        return int(depot)

    feas = unvisited_arr[feas_mask]
    m = len(feas)

    if m == 1:
        return int(feas[0])

    # Edge costs from current node to each feasible customer
    edge_costs = distance_matrix[current_node, feas]

    # Mean edge cost of the feasible set
    mean_edge = np.mean(edge_costs)

    # Dynamic regret-minimization score:
    # score_i = edge_cost_i - alpha * (edge_cost_i - mean_edge)
    #         = (1 - alpha) * edge_cost_i + alpha * mean_edge
    # Lower score is better.
    # With alpha in (0, 1), this dampens the effect of outlier edges.
    alpha = 0.3

    regret_scores = (1.0 - alpha) * edge_costs + alpha * mean_edge

    # Additionally, incorporate a savings-based bonus:
    # For each candidate j, the "savings" of inserting j after current vs going back to depot
    # is roughly: dist(current, depot) - dist(current, j) + dist(j, depot)
    # High savings means j is "worth visiting" relative to closing the route.
    # We use a small weight for this to encourage geometrically consistent routes.
    
    dist_to_depot = distance_matrix[feas, depot]
    dist_current_to_depot = distance_matrix[current_node, depot]
    
    # Savings heuristic component: prefer nodes where visiting them saves distance
    # savings_j = dist(current, depot) - dist(current, j) + dist(j, depot)
    # If savings > 0, visiting j is better than going straight to depot
    savings = dist_current_to_depot - edge_costs + dist_to_depot
    
    # Normalize savings to [0, 1] range for blending
    s_min = np.min(savings)
    s_max = np.max(savings)
    if s_max - s_min > 1e-9:
        savings_norm = (savings - s_min) / (s_max - s_min)
    else:
        savings_norm = np.zeros_like(savings)
    
    # Blend: primary signal is regret-minimized edge cost, secondary is savings
    # Lower combined score is better
    savings_weight = 0.2
    combined_scores = regret_scores - savings_weight * savings_norm * (mean_edge + 1e-9)

    # Select the candidate with the lowest combined score
    best_idx = np.argmin(combined_scores)
    
    return int(feas[best_idx])
# EVOLVE-BLOCK-END
