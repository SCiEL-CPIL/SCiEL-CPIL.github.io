"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_T0 = time.time()
_DEADLINE = _T0 + 145.0  # safety margin before 150s kill


def _time_ok():
    return (time.time() - _T0) < _DEADLINE


def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    n = len(unvisited_nodes)
    if n == 0:
        return int(depot)

    # Feasible: demand fits in remaining capacity
    unvisited_arr = np.asarray(unvisited_nodes, dtype=np.intp)
    feas_mask = demands[unvisited_arr] <= rest_capacity
    if not np.any(feas_mask):
        return int(depot)

    feas = unvisited_arr[feas_mask]
    k = len(feas)

    # Distance from current node to each feasible candidate
    d_cur = distance_matrix[current_node, feas]

    if k == 1:
        return int(feas[0])

    # Lookahead: for each candidate, distance to its nearest other feasible
    # neighbor (or to depot if none remain). This is a smooth surrogate for
    # "how expensive is the next step after I pick this node".
    # Build a small (k x k) sub-matrix among feasible nodes
    sub = distance_matrix[np.ix_(feas, feas)]
    # Distance to nearest OTHER feasible node (or depot fallback)
    # Set diagonal to inf so we don't pick self
    np.fill_diagonal(sub, np.inf)
    # If k==2, each candidate's nearest other is the other one; fine.
    # If a candidate has no other feasible neighbor (k==1 handled above),
    # we'd use depot distance, but k>=2 here so there's always another.
    nearest_other = sub.min(axis=1)

    # Smooth surrogate: score = d_cur + alpha * nearest_other
    # alpha=0.35 gives mild lookahead without overriding proximity
    alpha = 0.35
    scores = d_cur + alpha * nearest_other

    idx = int(np.argmin(scores))
    return int(feas[idx])
# EVOLVE-BLOCK-END
