"""Rewritten program for Capacitated Vehicle Routing; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_T0 = time.time()
_DEADLINE = _T0 + 148.0

def _time_ok():
    return (time.time() - _T0) < _DEADLINE

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    # Safety: if time is running out, close the route.
    if not _time_ok():
        return int(depot)

    if len(unvisited_nodes) == 0:
        return int(depot)

    unvisited_arr = np.asarray(unvisited_nodes, dtype=np.intp)

    # Capacity feasibility: node's demand must fit in remaining capacity.
    feas_mask = demands[unvisited_arr] <= rest_capacity
    if not np.any(feas_mask):
        return int(depot)

    feas_nodes = unvisited_arr[feas_mask]
    n_feas = feas_nodes.shape[0]

    # Distance slices (row vectors).
    d_cur = distance_matrix[current_node, feas_nodes]   # current -> candidate
    d_dep = distance_matrix[depot, feas_nodes]          # depot -> candidate

    # Precompute full sub-matrix among feasible nodes.
    D = distance_matrix[np.ix_(feas_nodes, feas_nodes)]   # n_feas x n_feas
    dem = demands[feas_nodes]
    rem = rest_capacity

    # Estimated marginal cost of choosing candidate i now using a 2-step lookahead.
    # For each candidate c:
    #   leg1 = d[cur, c]
    #   leg2 = min over feasible successors s of (d[c, s] + d[depot, s]) or d[c, depot]
    #   leg3 = 0 (already included in leg2's depot term)
    # Plus a savings/pull-back reward.

    leg2 = np.empty(n_feas, dtype=float)

    for i in range(n_feas):
        ci = dem[i]
        can_rem = rem - ci
        if can_rem <= 0:
            # No successor fits; must go to depot next
            leg2[i] = distance_matrix[feas_nodes[i], depot]
            continue
        succ_mask = (dem <= can_rem)
        succ_mask[i] = False
        if np.any(succ_mask):
            row = D[i, succ_mask]
            succ_dep = d_dep[succ_mask]
            # Cost if we go to successor j: d[c,s] + d[depot, s]
            combined = row + succ_dep
            # Also consider going straight to depot from c
            direct_to_depot = distance_matrix[feas_nodes[i], depot]
            # Take the minimum of the best successor option and direct depot
            best_succ_cost = np.min(combined)
            leg2[i] = min(best_succ_cost, direct_to_depot)
        else:
            leg2[i] = distance_matrix[feas_nodes[i], depot]

    # Savings-based pull-back: reward nodes that are far from depot but close to current.
    # The classic savings for inserting c is d[depot,c] + d[c,depot] - d[cur,c] - d[depot,cur]
    # But since d[depot,cur] is constant for all candidates, we focus on:
    #   pull = alpha * (d_dep - d_cur)  -- larger when c is far from depot but near current
    alpha = 0.35
    pull = alpha * np.clip(d_dep - d_cur, -1.0, None)

    # Total estimated cost
    total = d_cur + leg2 - pull

    # Tie-break: prefer smaller immediate distance, then smaller total
    order = np.lexsort((d_cur, total))
    best_idx = order[0]
    best_node = feas_nodes[best_idx]

    return int(best_node)
# EVOLVE-BLOCK-END
