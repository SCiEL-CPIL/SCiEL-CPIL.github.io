"""Capacitated vehicle routing: pick the next customer or return to depot."""
# EVOLVE-BLOCK-START
import time
import numpy as np

# Cache numpy views of the distance matrix to avoid repeated conversion cost.
_DM_CACHE = {}


def _get_dm(distance_matrix):
    key = id(distance_matrix)
    dm = _DM_CACHE.get(key)
    if dm is None:
        dm = np.asarray(distance_matrix)
        _DM_CACHE[key] = dm
    return dm


# Deadline guard: the evaluator kills the process at 150s wall-clock.
# We stop doing any work once we get close to that limit.
_T0 = time.perf_counter()
_DEADLINE = _T0 + 140.0  # 10s safety margin


def _deadline_exceeded():
    return time.perf_counter() > _DEADLINE


def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    # Safety: if we are about to be killed, just close the route quickly.
    if _deadline_exceeded():
        return int(depot)

    dm = _get_dm(distance_matrix)

    unvisited = np.asarray(unvisited_nodes, dtype=np.int64)
    n = unvisited.size

    if n == 0:
        return int(depot)

    # Base distance from current node back to depot (the "free" return cost).
    d_cur_depot = dm[current_node, depot]

    # Feasibility: candidate demand must fit in remaining capacity.
    cand_dem = np.asarray(demands)[unvisited]
    feasible_mask = cand_dem <= rest_capacity

    # If nothing fits, we must return to depot.
    if not np.any(feasible_mask):
        return int(depot)

    if np.count_nonzero(feasible_mask) == n:
        cand = unvisited
        cand_dem = cand_dem
    else:
        cand = unvisited[feasible_mask]
        cand_dem = cand_dem[feasible_mask]

    # Heuristic: Lookahead Nearest Neighbor with Demand Bias
    # Score = dist(current, cand) + min(dist(cand, next_feasible), dist(cand, depot)) - bias * demand
    
    d_cur_cand = dm[current_node, cand]
    d_cand_depot = dm[cand, depot]
    
    k = len(cand)
    if k > 1:
        # Distance matrix between candidates
        cand_dm = dm[np.ix_(cand, cand)]
        # Set diagonal to infinity to exclude self-distance
        np.fill_diagonal(cand_dm, np.inf)
        min_dist_to_other = np.min(cand_dm, axis=1)
        min_next_cost = np.minimum(min_dist_to_other, d_cand_depot)
    else:
        # Only one candidate, next step must be depot
        min_next_cost = d_cand_depot
        
    # Bias: prefer larger demands to fill the vehicle
    # Weight 0.1 is a guess; scales with demand (1-9) to affect score
    score = d_cur_cand + min_next_cost - 0.1 * cand_dem

    best_idx = int(np.argmin(score))
    return int(cand[best_idx])
# EVOLVE-BLOCK-END
