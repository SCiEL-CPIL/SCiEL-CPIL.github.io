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
      3. Score each feasible node using a lookahead metric:
         - immediate distance from current_node
         - plus a fraction of the minimum distance from that node to any other feasible node or depot
         This avoids dead-end picks where a node is close now but far from everything else.
    """
    if _time_exceeded():
        # Safety: return depot to close route quickly
        return 0

    unvisited = list(unvisited_nodes)
    if not unvisited:
        return 0

    # Filter by capacity
    feasible = [n for n in unvisited if demands[n] <= rest_capacity]
    if not feasible:
        return 0

    # Convert to numpy arrays for vectorized operations
    feasible_arr = np.array(feasible)
    dist_from_current = distance_matrix[current_node, feasible_arr]

    # Lookahead: for each feasible node, compute a soft-min distance to any other
    # feasible node or depot. We want to avoid nodes that are isolated (far from
    # everything else). The soft-min (log-sum-exp) is a smooth surrogate for the
    # hard min, giving a smoother objective surface than the nonsmooth np.min.
    n_feasible = len(feasible_arr)
    if n_feasible == 1:
        return int(feasible_arr[0])

    # Build a small distance matrix among feasible nodes + depot
    # indices: feasible_arr plus depot
    indices = np.concatenate([feasible_arr, [depot]])
    sub_dist = distance_matrix[np.ix_(indices, indices)]

    # Vectorized soft-min over each row of sub_dist, excluding the self entry.
    # softmin(x) = -t * log( sum_i exp(-x_i / t) ), which approaches min(x) as t -> 0
    # and is smooth in all entries.
    t = 3.0
    # Mask diagonal (self) with a large value so it doesn't contribute
    diag_mask = np.eye(n_feasible + 1, dtype=bool)
    masked_sub = sub_dist.copy()
    masked_sub[diag_mask] = 1e9
    
    # Numerical stability: subtract row-wise min of finite values
    row_min = np.min(masked_sub, axis=1)
    shifted = masked_sub - row_min[:, np.newaxis]
    exp_vals = np.exp(-shifted / t)
    # Zero out the diagonal contribution (self)
    exp_vals[diag_mask] = 0.0
    s = np.sum(exp_vals, axis=1)
    # softmin = row_min + t * log(s)
    lookahead = row_min[:n_feasible] + t * np.log(s[:n_feasible])
    
    # Score: immediate distance + alpha * lookahead
    # A smaller alpha trusts the immediate distance more; the soft lookahead
    # gently penalizes dead-end picks without overwhelming the near-term cost.
    alpha = 0.4
    scores = dist_from_current + alpha * lookahead

    # Pick the node with minimum score
    best_idx = np.argmin(scores)
    return int(feasible_arr[best_idx])
# EVOLVE-BLOCK-END
