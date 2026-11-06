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
      3. Use a savings-style route extension:
         - candidate cost to insert/extend from current_node to node i
         - reward edges with large savings relative to direct depot return
         - penalize nodes that leave poor continuation options
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

    feasible_arr = np.array(feasible)
    n_feasible = len(feasible_arr)
    if n_feasible == 1:
        return int(feasible_arr[0])

    # Distances from current node and depot to feasible nodes
    d_cur = distance_matrix[current_node, feasible_arr]
    d_dep = distance_matrix[depot, feasible_arr]

    # Small matrix among feasible nodes
    sub = distance_matrix[np.ix_(feasible_arr, feasible_arr)]

    # For each feasible node i:
    #   base cost: go from current to i, then likely return to depot
    #   savings: if we visit i, compare (cur->i + i->dep) vs (cur->dep + i->dep)
    #            larger savings means better insertion into current route
    cur_to_dep = distance_matrix[current_node, depot]
    savings = cur_to_dep + d_dep - d_cur - d_dep  # simplifies to cur_to_dep - d_cur
    # But more meaningful: edge savings for i relative to nearest continuation j:
    # We want to favor i if it has a cheap outgoing edge to another feasible node or depot.
    # Compute min outgoing distance from i to any other feasible node or depot.
    # Build extended matrix: feasible nodes + depot row/col
    ext = np.empty((n_feasible + 1, n_feasible + 1), dtype=float)
    ext[:n_feasible, :n_feasible] = sub
    ext[:n_feasible, n_feasible] = d_dep
    ext[n_feasible, :n_feasible] = d_dep
    ext[n_feasible, n_feasible] = 0.0

    # Vectorized soft-min outgoing cost (excluding self)
    # Use log-sum-exp style soft min with temperature
    T = 2.0
    # For each i, softmin over j != i of ext[i, j]
    # Mask diagonal for feasible rows
    mask = ~np.eye(n_feasible + 1, dtype=bool)
    masked = np.where(mask, ext, np.inf)
    # Soft min: -T * log(sum(exp(-ext/T)))
    # To avoid overflow, subtract min per row
    row_min = np.min(masked, axis=1)
    exp_terms = np.exp(-(masked - row_min[:, None]) / T)
    exp_terms = np.where(mask, exp_terms, 0.0)
    softmin = row_min - T * np.log(np.sum(exp_terms, axis=1) + 1e-12)

    # Score:
    # - immediate travel from current to i
    # - expected return/continuation cost after i (softmin)
    # - small penalty for low savings (i.e., prefer nodes that reduce depot-return cost)
    # We use lower score = better.
    w_immediate = 1.0
    w_continuation = 0.55
    w_savings = -0.35  # negative weight: larger savings lowers score

    scores = (
        w_immediate * d_cur
        + w_continuation * softmin[:n_feasible]
        + w_savings * savings
    )

    best_idx = int(np.argmin(scores))
    return int(feasible_arr[best_idx])
# EVOLVE-BLOCK-END
