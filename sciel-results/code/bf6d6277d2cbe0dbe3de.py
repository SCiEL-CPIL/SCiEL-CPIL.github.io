"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global deadline guard: stop doing heavy work if we are too close to the 120s limit
_DEADLINE = time.time() + 115.0  # 5s safety margin before the 120s kill


def _time_exceeded():
    return time.time() > _DEADLINE


def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Heuristic: regret-based 2-step lookahead.
    
    For each feasible candidate c:
      - cost_if_pick_c = dist(current, c) + min( dist(c, d) for d in feasible \ {c} )
                         (if only one feasible, use dist(c, depot))
      - cost_if_skip_c = dist(current, best_other) + min( dist(best_other, e) for e in feasible \ {best_other} )
                         where best_other is the cheapest next node when c is excluded
    regret(c) = cost_if_skip_c - cost_if_pick_c
    Pick c with maximum regret.
    
    This directly targets the incremental cost of the decision, rather than a weighted sum.
    """
    if len(unvisited_nodes) == 0:
        return depot

    # Filter by capacity
    feasible = [i for i in unvisited_nodes if demands[i] <= rest_capacity]
    if len(feasible) == 0:
        return depot
    if len(feasible) == 1:
        return int(feasible[0])

    # Time guard: if we are running out of time, fall back to simple nearest
    if _time_exceeded():
        idx = np.argmin(distance_matrix[current_node, feasible])
        return int(feasible[idx])

    n = len(feasible)
    dm = distance_matrix

    # Precompute distance from current to each feasible node
    dist_from_current = dm[current_node, feasible]

    # If n is small enough, do full regret computation
    # For larger n, use a simplified regret
    if n <= 30:
        # Build sub-distance matrix among feasible nodes
        feas_arr = np.array(feasible)
        sub_dm = dm[np.ix_(feas_arr, feas_arr)]
        # Also need distance from each feasible to depot
        dist_to_depot = dm[feas_arr, depot]

        # For each candidate i, compute:
        # cost_if_pick_i = dist_from_current[i] + min_j!=i (sub_dm[i,j])  [or dist_to_depot[i] if n==2]
        # cost_if_skip_i = dist_from_current[best_j] + min_k!=j (sub_dm[j,k])
        #   where best_j = argmin_k (dist_from_current[k]) for k != i

        # Precompute min distance from each feasible node to any other feasible node
        # (excluding self)
        sub_dm_copy = sub_dm.copy()
        np.fill_diagonal(sub_dm_copy, np.inf)
        min_to_other = np.min(sub_dm_copy, axis=1)

        # cost_if_pick[i] = dist_from_current[i] + min_to_other[i]
        cost_if_pick = dist_from_current + min_to_other

        # For cost_if_skip: for each i, find the best j != i by dist_from_current,
        # then cost = dist_from_current[j] + min_to_other[j]
        # We can precompute the top-2 nearest by dist_from_current
        sorted_indices = np.argsort(dist_from_current)
        # For each i, the best j != i is:
        #   if sorted_indices[0] != i, then j = sorted_indices[0]
        #   else j = sorted_indices[1]
        best_j_for_skip = np.empty(n, dtype=int)
        for i in range(n):
            if sorted_indices[0] != i:
                best_j_for_skip[i] = sorted_indices[0]
            else:
                best_j_for_skip[i] = sorted_indices[1]

        cost_if_skip = dist_from_current[best_j_for_skip] + min_to_other[best_j_for_skip]

        regret = cost_if_skip - cost_if_pick
        best_i = np.argmax(regret)
        return int(feasible[best_i])
    else:
        # For larger n, use a cheaper approximation:
        # regret approx = (dist_from_current[best_other] - dist_from_current[i]) 
        #                  + (min_to_other[best_other] - min_to_other[i])
        # where best_other is the nearest to current (excluding i)
        feas_arr = np.array(feasible)
        sub_dm = dm[np.ix_(feas_arr, feas_arr)]
        sub_dm_copy = sub_dm.copy()
        np.fill_diagonal(sub_dm_copy, np.inf)
        min_to_other = np.min(sub_dm_copy, axis=1)

        sorted_indices = np.argsort(dist_from_current)
        best_j_for_skip = np.empty(n, dtype=int)
        for i in range(n):
            if sorted_indices[0] != i:
                best_j_for_skip[i] = sorted_indices[0]
            else:
                best_j_for_skip[i] = sorted_indices[1]

        regret = (dist_from_current[best_j_for_skip] - dist_from_current) + \
                 (min_to_other[best_j_for_skip] - min_to_other)
        best_i = np.argmax(regret)
        return int(feasible[best_i])
# EVOLVE-BLOCK-END
