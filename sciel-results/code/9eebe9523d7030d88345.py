"""Regret-based heuristic for Capacitated VRP with capacity awareness."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = None
_TIME_LIMIT = 118.0  # Safety margin before 120s kill


def _check_deadline():
    """Return True if we should stop and return depot for safety."""
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return (time.time() - _start_time) > _TIME_LIMIT


def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Regret-based selection: score each feasible customer by the difference
    between its best and second-best connection costs, prioritizing nodes
    that are uniquely well-positioned.
    
    Returns 0 (depot) to close route if no feasible customers or deadline exceeded.
    """
    # Deadline guard
    if _check_deadline():
        return 0
    
    if len(unvisited_nodes) == 0:
        return 0
    
    unvisited_arr = np.array(unvisited_nodes)
    
    # Filter to only feasible customers (demand <= rest_capacity)
    feasible_mask = demands[unvisited_arr] <= rest_capacity
    feasible_nodes = unvisited_arr[feasible_mask]
    
    if len(feasible_nodes) == 0:
        return 0
    
    n_feasible = len(feasible_nodes)
    
    # If only one feasible node, return it
    if n_feasible == 1:
        return int(feasible_nodes[0])
    
    # Get distances from current node to feasible customers
    dist_from_current = distance_matrix[current_node, feasible_nodes]
    
    # Get distances from feasible customers back to depot
    dist_to_depot = distance_matrix[feasible_nodes, depot]
    
    # Get distances between all pairs of feasible customers
    # This is needed for regret calculation
    pair_dists = distance_matrix[np.ix_(feasible_nodes, feasible_nodes)]
    
    demands_feasible = demands[feasible_nodes]
    
    # Base score components:
    # 1. Direct distance from current node
    # 2. Return to depot distance (weighted)
    # 3. Regret: difference between best and second-best "connection quality"
    
    # For each feasible node i, compute:
    # - Best alternative: the other feasible node j that gives the lowest
    #   total cost if we visited j instead of i
    # - The regret is how much worse it would be to not visit i now
    
    # Simplified regret: for node i, consider the cost of going current->i->depot
    # vs current->j->depot for the best j != i
    # Regret = min_j(cost_j) - cost_i, where cost_i = dist_current_i + w*dist_i_depot
    
    lambda_return = 0.5
    
    # Cost if we visit node i and then return to depot (immediate closure)
    closure_cost = dist_from_current + lambda_return * dist_to_depot
    
    # For regret, we want to know: if we don't pick node i now, what's the
    # best alternative? The regret is high if node i is uniquely good.
    
    # Compute for each node i: the minimum closure_cost among all other nodes
    # This is the "opportunity cost" of not picking i
    
    # To do this efficiently:
    # Sort closure_cost, then for each position, the min of others is either
    # the global min (if i is not the argmin) or the second-min (if i is the argmin)
    
    min_cost = np.min(closure_cost)
    min_idx = np.argmin(closure_cost)
    
    # Second minimum
    sorted_costs = np.sort(closure_cost)
    if n_feasible >= 2:
        second_min_cost = sorted_costs[1]
    else:
        second_min_cost = min_cost
    
    # Regret for node i:
    # If i is the best, regret = min_cost - second_min_cost (how much better i is)
    # If i is not the best, regret = 0 (we have a better option)
    
    # Actually, a more useful regret: for each node i, what is the cost of
    # the best alternative to visiting i? If we skip i, we'll have to visit
    # some other node. The regret is high if all alternatives are much worse.
    
    # Alternative approach: regret = min_{j != i} closure_cost[j] - closure_cost[i]
    # This is negative if i is not the best, positive if i is the best.
    # We want to maximize this (or minimize the negative).
    
    # Let's compute: for each i, the minimum closure cost among j != i
    # This can be done by: if i == min_idx, use second_min_cost; else use min_cost
    
    min_others = np.full(n_feasible, min_cost)
    min_others[min_idx] = second_min_cost
    
    # Regret: how much better is the best alternative compared to i?
    # High regret means i is uniquely good (best alternative is much worse)
    regret = min_others - closure_cost
    
    # Combine: we want low closure_cost AND high regret
    # Score = closure_cost - alpha * regret
    # Lower score is better
    
    alpha = 0.3
    
    # Capacity efficiency: prefer nodes with higher demand relative to rest_capacity
    # when distances are comparable
    cap_ratio = demands_feasible / float(rest_capacity)
    
    # Normalize closure_cost for scaling
    max_closure = np.max(closure_cost)
    if max_closure > 0:
        closure_norm = closure_cost / (max_closure + 1e-9)
    else:
        closure_norm = closure_cost
    
    # Small penalty for low capacity utilization
    cap_penalty = 0.1 * (1 - cap_ratio) * closure_norm
    
    # Final score: lower is better
    scores = closure_cost - alpha * regret + cap_penalty
    
    # Pick the customer with minimum score
    best_idx = np.argmin(scores)
    return int(feasible_nodes[best_idx])
# EVOLVE-BLOCK-END
