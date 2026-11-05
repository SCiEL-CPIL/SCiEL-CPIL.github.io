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

    # Time check
    if not _time_ok():
        # Fallback to nearest feasible or depot if time is up
        unvisited_arr = np.asarray(unvisited_nodes, dtype=np.intp)
        feas_mask = demands[unvisited_arr] <= rest_capacity
        if not np.any(feas_mask):
            return int(depot)
        feas = unvisited_arr[feas_mask]
        if len(feas) == 0:
            return int(depot)
        return int(feas[np.argmin(distance_matrix[current_node, feas])])

    unvisited_arr = np.asarray(unvisited_nodes, dtype=np.intp)
    
    # Feasibility: demand must fit in remaining capacity
    feas_mask = demands[unvisited_arr] <= rest_capacity
    if not np.any(feas_mask):
        return int(depot)

    feas = unvisited_arr[feas_mask]
    m = len(feas)
    
    if m == 1:
        return int(feas[0])

    # Dynamic regret-minimization with soft forward-look:
    # 1) Regret term: d(current, j) - mean(d(current, feas)) -> avoid outlier-long edges
    # 2) Forward-look: soft-min over d(j, k) for k in feas\{j} -> prefer nodes that
    #    keep cheap options open for the next step (log-sum-exp smoothing)
    d_current = distance_matrix[current_node, feas]
    mean_d = np.mean(d_current)

    # Regret score (lower = better)
    scores = d_current - mean_d

    # Forward-look: for each candidate j, compute soft-min of distances to other feasible nodes
    # Use log-sum-exp for a smooth approximation of min
    if m > 1:
        # d_feas[i, k] = distance_matrix[feas[i], feas[k]]
        d_feas = distance_matrix[np.ix_(feas, feas)]
        # For each i, we want soft-min over k != i of d_feas[i, k]
        # Set diagonal to a large value to exclude self
        d_feas_diag = d_feas.copy()
        np.fill_diagonal(d_feas_diag, 1e9)
        # Log-sum-exp for soft-min: -logsumexp(-x) approximates min(x)
        # soft_min_i = -logsumexp(-d_feas_diag[i, :])
        # Use stable logsumexp: max_val + log(sum(exp(x - max_val)))
        neg_d = -d_feas_diag
        max_neg = np.max(neg_d, axis=1, keepdims=True)
        log_sum_exp = max_neg.squeeze() + np.log(np.sum(np.exp(neg_d - max_neg), axis=1))
        soft_min_forward = -log_sum_exp  # shape (m,)
        # Scale forward-look to be comparable with regret term
        # The forward-look ranges roughly [0, max_dist], regret ranges [-max, max]
        # Use a small weight to make it a tiebreaker / secondary criterion
        scores = scores + 0.3 * soft_min_forward

    idx = np.argmin(scores)
    return int(feas[idx])
# EVOLVE-BLOCK-END
