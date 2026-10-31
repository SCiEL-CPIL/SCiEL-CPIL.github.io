"""MST-aware TSP constructive heuristic with per-instance parameter fitting.

Strategy:
  1. For each new instance, fit parameters (alpha, gamma, delta) via a small
     grid search using a fast approximate tour builder.
  2. Score each candidate c as:
       score(c) = d(cur,c) + alpha*d(c,dest)
                  + gamma * MST_weight(unvisited \ {c})
                  + delta * (d(c,dest) - min_u d(u,dest))
     The MST weight of the remaining subgraph is a principled lower bound on
     the cost to connect all remaining nodes, so minimizing it preserves
     low-cost connectivity for the residual tour.
  3. Reset per-instance state when a new instance is detected.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_LIMIT = 115.0  # Safety margin before 120s limit

# Per-instance state
_current_instance_id = -1
_FIT_PARAMS = None  # (alpha, gamma, delta) best found by fitting
_instance_counter = 0
_last_dm_checksum = None


def _is_deadline_passed():
    return (time.time() - _start_time) > _LIMIT


def _mst_weight(sub_dist):
    """Compute the MST weight of a symmetric distance matrix using Prim's
    algorithm (vectorized inner loop). Returns 0 for empty/1-node graphs."""
    n = sub_dist.shape[0]
    if n <= 1:
        return 0.0
    in_mst = np.zeros(n, dtype=bool)
    min_edge = sub_dist[0].copy()
    in_mst[0] = True
    total = 0.0
    for _ in range(n - 1):
        # Mask out already-in-MST nodes
        min_edge[in_mst] = np.inf
        idx = int(np.argmin(min_edge))
        total += min_edge[idx]
        in_mst[idx] = True
        # Update min_edge with distances from the newly added node
        new_dists = sub_dist[idx]
        mask = ~in_mst
        min_edge[mask] = np.minimum(min_edge[mask], new_dists[mask])
    return float(total)


def _build_tour_fast(distance_matrix, start_node, alpha=0.0, gamma=0.0,
                     delta=0.0):
    """Build a full TSP tour using the given parameters. Returns tour length.

    Uses MST-based residual cost for the gamma term.
    """
    n = distance_matrix.shape[0]
    current = start_node
    total_dist = 0.0
    remaining = list(range(1, n))  # 0 is start

    while len(remaining) > 0:
        rem_arr = np.array(remaining)
        k = len(rem_arr)

        if k == 1:
            next_node = int(rem_arr[0])
            total_dist += distance_matrix[current, next_node]
            break

        # Vectorized scoring
        d_cur = distance_matrix[current, rem_arr]
        scores = d_cur.copy()

        if alpha > 0:
            d_dest = distance_matrix[rem_arr, start_node]
            scores += alpha * d_dest

        if gamma > 0 and k > 2:
            # For each candidate c, compute MST weight of remaining \ {c}
            # This is expensive (O(k * k^2)), but k is small early on and
            # the grid search uses a limited number of evaluations.
            # Optimization: only compute MST for a subset if k is large
            if k <= 30:
                mst_costs = np.zeros(k)
                for i in range(k):
                    # Build submatrix of remaining without candidate i
                    mask = np.ones(k, dtype=bool)
                    mask[i] = False
                    sub_idx = rem_arr[mask]
                    sub_dist = distance_matrix[np.ix_(sub_idx, sub_idx)]
                    mst_costs[i] = _mst_weight(sub_dist)
                scores += gamma * mst_costs
            else:
                # For large k, use a cheaper proxy: sum of k-1 smallest
                # outgoing edges from each node (lower bound on MST)
                sub_dist = distance_matrix[np.ix_(rem_arr, rem_arr)].copy()
                np.fill_diagonal(sub_dist, np.inf)
                # For each row, sum of the (k-2) smallest values (since we
                # remove one node, the remaining graph has k-1 nodes, and
                # MST has k-2 edges)
                # Approximate: use min edge + second min edge as proxy
                sorted_rows = np.sort(sub_dist, axis=1)
                # MST lower bound ~ sum of smallest (k-2) edges per node / 2
                # But that's not quite right. Use: min outgoing edge as proxy
                # for connectivity cost.
                min_out = np.min(sub_dist, axis=1)
                scores += gamma * min_out

        if delta > 0 and k > 2:
            d_dest_all = distance_matrix[rem_arr, start_node]
            min_d_dest = np.min(d_dest_all)
            scores += delta * (d_dest_all - min_d_dest)

        best_idx = int(np.argmin(scores))
        next_node = int(rem_arr[best_idx])
        total_dist += distance_matrix[current, next_node]
        remaining.pop(best_idx)
        current = next_node

    # Close the tour
    total_dist += distance_matrix[current, start_node]
    return total_dist


def _fit_params(distance_matrix, start_node, time_budget=2.5):
    """Fit alpha, gamma, delta by grid search on the full instance."""
    global _FIT_PARAMS

    if _is_deadline_passed():
        return (0.0, 0.0, 0.0)

    # Focused grid
    alpha_values = [0.0, 0.05, 0.1, 0.15, 0.2]
    gamma_values = [0.0, 0.1, 0.2, 0.3, 0.5]
    delta_values = [0.0, 0.05, 0.1]

    best_alpha = 0.0
    best_gamma = 0.0
    best_delta = 0.0
    best_len = float('inf')

    elapsed_start = time.time()
    remaining_time = _LIMIT - (time.time() - _start_time) - 5.0
    deadline = elapsed_start + min(time_budget, max(0.5, remaining_time))

    for alpha in alpha_values:
        if time.time() > deadline:
            break
        for gamma in gamma_values:
            if time.time() > deadline:
                break
            for delta in delta_values:
                if time.time() > deadline:
                    break
                tour_len = _build_tour_fast(distance_matrix, start_node,
                                            alpha, gamma, delta)
                if tour_len < best_len:
                    best_len = tour_len
                    best_alpha = alpha
                    best_gamma = gamma
                    best_delta = delta

    _FIT_PARAMS = (best_alpha, best_gamma, best_delta)
    return (best_alpha, best_gamma, best_delta)


def _detect_new_instance(distance_matrix, unvisited_nodes):
    """Detect if we're starting a new instance."""
    global _last_dm_checksum

    n_total = distance_matrix.shape[0]
    n_unvisited = len(unvisited_nodes)

    if n_unvisited >= n_total - 1:
        checksum = np.sum(distance_matrix[::max(1, n_total // 10),
                                          ::max(1, n_total // 10)])
        if _last_dm_checksum is None or abs(checksum - _last_dm_checksum) > 1e-6:
            _last_dm_checksum = checksum
            return True
        elif distance_matrix.shape != (n_total, n_total):
            _last_dm_checksum = checksum
            return True
    return False


def select_next_node(current_node, destination_node, unvisited_nodes,
                     distance_matrix):
    """
    Select the next node to visit in the TSP tour.

    Uses MST-based residual cost estimation with per-instance parameter
    fitting. The score for candidate c is:
      d(cur,c) + alpha*d(c,dest) + gamma*MST(unvisited\{c})
                + delta*(d(c,dest) - min_u d(u,dest))
    """
    global _FIT_PARAMS, _instance_counter, _last_dm_checksum

    n_unvisited = len(unvisited_nodes)

    if n_unvisited == 0:
        return -1

    if n_unvisited == 1:
        return int(unvisited_nodes[0])

    # Detect new instance
    is_new_instance = _detect_new_instance(distance_matrix, unvisited_nodes)

    if is_new_instance:
        _instance_counter += 1
        _FIT_PARAMS = None
        tb = 2.5 if _instance_counter == 1 else 1.0
        _FIT_PARAMS = _fit_params(distance_matrix, current_node,
                                  time_budget=tb)

    if _FIT_PARAMS is None:
        alpha, gamma, delta = 0.1, 0.2, 0.05
    else:
        alpha, gamma, delta = _FIT_PARAMS

    # If deadline is close, use pure nearest neighbor
    if _is_deadline_passed():
        dists_from_current = distance_matrix[current_node, unvisited_nodes]
        best_idx = np.argmin(dists_from_current)
        return int(unvisited_nodes[best_idx])

    # Compute scores for candidates (vectorized)
    dists_from_current = distance_matrix[current_node, unvisited_nodes]
    scores = dists_from_current.copy()

    if alpha > 0:
        dists_to_dest = distance_matrix[unvisited_nodes, destination_node]
        scores += alpha * dists_to_dest

    if gamma > 0 and n_unvisited > 2:
        # MST-based residual cost: for each candidate c, compute MST weight
        # of the subgraph induced by unvisited \ {c}
        unvisited_arr = np.asarray(unvisited_nodes)
        k = len(unvisited_arr)

        if k <= 35:
            # Full MST computation for each candidate (expensive but accurate)
            mst_costs = np.zeros(k)
            for i in range(k):
                mask = np.ones(k, dtype=bool)
                mask[i] = False
                sub_idx = unvisited_arr[mask]
                sub_dist = distance_matrix[np.ix_(sub_idx, sub_idx)]
                mst_costs[i] = _mst_weight(sub_dist)
            scores += gamma * mst_costs
        else:
            # Cheaper proxy for large k: use the minimum outgoing edge
            # from each node as a connectivity cost estimate
            sub_dist = distance_matrix[np.ix_(unvisited_arr, unvisited_arr)].copy()
            np.fill_diagonal(sub_dist, np.inf)
            min_out = np.min(sub_dist, axis=1)
            scores += gamma * min_out

    if delta > 0 and n_unvisited > 2:
        d_dest_all = distance_matrix[unvisited_nodes, destination_node]
        min_d_dest = np.min(d_dest_all)
        scores += delta * (d_dest_all - min_d_dest)

    best_idx = np.argmin(scores)
    return int(unvisited_nodes[best_idx])

# EVOLVE-BLOCK-END
