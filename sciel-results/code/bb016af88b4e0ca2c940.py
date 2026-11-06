"""Combined program: savings-based primary with soft-min lookahead and capacity bonus."""
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
    
    Strategy (combined):
      1. Filter unvisited nodes that fit in rest_capacity.
      2. If none fit, return depot.
      3. Score each feasible node using:
         - savings (primary, from Clarke-Wright): dist(depot,cur)+dist(depot,j)-dist(cur,j)
           Higher savings = less marginal cost. We minimize -savings.
         - soft-min lookahead: smooth surrogate for min distance to any other feasible
           node or depot (penalizes dead-ends).
         - capacity utilization bonus: prefer nodes that use more remaining capacity.
         - small proximity tie-break.
    """
    if _time_exceeded():
        return 0

    unvisited = np.asarray(unvisited_nodes, dtype=np.int64)
    if len(unvisited) == 0:
        return 0

    # Filter by capacity
    feasible_mask = demands[unvisited] <= rest_capacity
    if not np.any(feasible_mask):
        return 0

    feasible_arr = unvisited[feasible_mask]
    n_feasible = len(feasible_arr)
    if n_feasible == 1:
        return int(feasible_arr[0])

    # --- Primary: savings (Clarke-Wright style) ---
    # savings[j] = dist(depot, current) + dist(depot, j) - dist(current, j)
    # Higher savings is better. We minimize negative savings.
    dist_depot_current = distance_matrix[depot, current_node]
    dist_depot_feasible = distance_matrix[depot, feasible_arr]
    dist_from_current = distance_matrix[current_node, feasible_arr]

    savings = dist_depot_current + dist_depot_feasible - dist_from_current
    # Convert to minimization: we want to minimize -savings
    neg_savings = -savings

    # --- Secondary: soft-min lookahead ---
    # For each feasible node, compute smooth min distance to any other feasible node or depot.
    # Penalizes dead-ends (nodes far from everything else).
    indices = np.concatenate([feasible_arr, [depot]])
    sub_dist = distance_matrix[np.ix_(indices, indices)]

    t = 2.0  # temperature for soft-min (lower = sharper, closer to hard min)
    diag_mask = np.eye(n_feasible + 1, dtype=bool)
    masked_sub = sub_dist.copy()
    masked_sub[diag_mask] = 1e9

    # Numerical stability
    row_min = np.min(masked_sub, axis=1)
    shifted = masked_sub - row_min[:, np.newaxis]
    exp_vals = np.exp(-shifted / t)
    exp_vals[diag_mask] = 0.0
    s = np.sum(exp_vals, axis=1)
    # Avoid log(0)
    s = np.maximum(s, 1e-300)
    lookahead = row_min[:n_feasible] + t * np.log(s[:n_feasible])

    # --- Tertiary: capacity utilization bonus ---
    # Prefer nodes that consume more of the remaining capacity (fuller trucks)
    cap_bonus = -0.08 * (demands[feasible_arr] / rest_capacity)

    # --- Small proximity tie-break ---
    # Normalize dist_from_current to [0,1] for a gentle tie-break
    dist_min = np.min(dist_from_current)
    dist_max = np.max(dist_from_current)
    dist_range = dist_max - dist_min + 1e-10
    dist_norm = (dist_from_current - dist_min) / dist_range

    # --- Combined score (minimize) ---
    # Weights: savings dominant, lookahead secondary, cap bonus and proximity small
    alpha_savings = 1.0       # primary
    alpha_lookahead = 0.25    # secondary
    alpha_cap = 1.0           # cap_bonus already scaled
    alpha_dist = 0.05         # gentle proximity tie-break

    scores = (alpha_savings * neg_savings
              + alpha_lookahead * lookahead
              + alpha_cap * cap_bonus
              + alpha_dist * dist_norm)

    best_idx = np.argmin(scores)
    return int(feasible_arr[best_idx])
# EVOLVE-BLOCK-END
