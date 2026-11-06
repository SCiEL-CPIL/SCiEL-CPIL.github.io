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
      3. Score each feasible node using a combined metric:
         - immediate distance from current_node
         - savings penalty (extra cost vs going straight to depot)
         - soft-min lookahead: smooth surrogate for min distance to any other feasible node or depot
         - capacity utilization bonus: encourage fuller trucks
      This balances proximity, route efficiency, avoids dead-end picks, and promotes capacity usage.
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
    dist_to_depot = distance_matrix[feasible_arr, depot]
    dist_current_to_depot = distance_matrix[current_node, depot]
    savings_penalty = dist_from_current + dist_to_depot - dist_current_to_depot

    # Vectorized soft-min lookahead (1-step): for each feasible node, compute a smooth
    # surrogate for the minimum distance to any other feasible node or depot.
    indices = np.concatenate([feasible_arr, [depot]])
    sub_dist = distance_matrix[np.ix_(indices, indices)]

    t = 1.8  # temperature for soft-min (lower = sharper, closer to hard min)
    diag_mask = np.eye(n_feasible + 1, dtype=bool)
    masked_sub = sub_dist.copy()
    masked_sub[diag_mask] = 1e9

    # Numerical stability: subtract row-wise min
    row_min = np.min(masked_sub, axis=1)
    shifted = masked_sub - row_min[:, np.newaxis]
    exp_vals = np.exp(-shifted / t)
    exp_vals[diag_mask] = 0.0
    s = np.sum(exp_vals, axis=1)
    lookahead_1 = row_min[:n_feasible] + t * np.log(s[:n_feasible])

    # 2-step soft-min lookahead: after going to node i, what's the best next hop?
    # For each feasible node i, compute soft-min over j (j != i) of
    #   dist(i, j) + soft_min_k(dist(j, k) + dist(k, depot))
    # This is expensive, so we approximate: use the 1-step lookahead from j to
    # estimate the cost of continuing from j.
    # cost_2step[i] = softmin_j ( dist(i,j) + lookahead_1[j] + dist(j, depot) )
    # where j ranges over feasible nodes + depot
    # We already have lookahead_1 for feasible nodes; for depot, lookahead is 0 (we're done).
    
    # Build a matrix: rows = feasible nodes, cols = feasible nodes + depot
    # entry (i, j) = dist(i, j) + future_cost(j)
    # future_cost(depot) = 0, future_cost(node_j) = lookahead_1[j] + dist(j, depot)
    
    # dist from feasible nodes to (feasible nodes + depot)
    dist_i_to_all = sub_dist[:n_feasible, :]  # shape (n_feasible, n_feasible+1)
    
    # future cost for each column
    future_cost = np.zeros(n_feasible + 1)
    future_cost[:n_feasible] = lookahead_1 + dist_to_depot  # dist(j, depot)
    # future_cost[depot_idx] = 0 already
    
    # total cost matrix: (n_feasible, n_feasible+1)
    cost_matrix = dist_i_to_all + future_cost[np.newaxis, :]
    
    # Mask diagonal (can't go to self)
    cost_matrix[np.arange(n_feasible), np.arange(n_feasible)] = 1e9
    
    # Soft-min over columns for each row
    t2 = 1.5
    row_min2 = np.min(cost_matrix, axis=1)
    shifted2 = cost_matrix - row_min2[:, np.newaxis]
    exp_vals2 = np.exp(-shifted2 / t2)
    s2 = np.sum(exp_vals2, axis=1)
    lookahead_2 = row_min2 + t2 * np.log(s2)

    # Capacity utilization bonus: prefer nodes that use more of the remaining capacity
    # This encourages fuller trucks, reducing total number of routes
    cap_bonus = -0.04 * (demands[feasible_arr] / rest_capacity)

    # Combined score
    alpha_savings = 0.22
    alpha_lookahead_1 = 0.20
    alpha_lookahead_2 = 0.25
    scores = dist_from_current + alpha_savings * savings_penalty + alpha_lookahead_1 * lookahead_1 + alpha_lookahead_2 * lookahead_2 + cap_bonus

    best_idx = np.argmin(scores)
    return int(feasible_arr[best_idx])
# EVOLVE-BLOCK-END
