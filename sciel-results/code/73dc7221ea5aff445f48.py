"""CVRP heuristic: savings-based scoring with capacity pressure and lookahead."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE = 150.0
_START_TIME = time.perf_counter()
_MARGIN = 5.0


def _elapsed():
    return time.perf_counter() - _START_TIME


def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """Pick next customer or return to depot (0).
    
    Combines Clarke-Wright savings with capacity pressure and 1-step lookahead.
    """
    # Safety guard
    if _elapsed() > _DEADLINE - _MARGIN:
        feasible = [n for n in unvisited_nodes if demands[n] <= rest_capacity]
        if not feasible:
            return int(depot)
        return int(feasible[int(np.argmin(distance_matrix[current_node, feasible]))])

    unvisited = np.asarray(unvisited_nodes, dtype=int)
    if unvisited.size == 0:
        return int(depot)

    demands_arr = np.asarray(demands)
    dist = np.asarray(distance_matrix)

    # Feasibility mask
    feasible_mask = demands_arr[unvisited] <= rest_capacity
    if not feasible_mask.any():
        return int(depot)

    feasible = unvisited[feasible_mask]
    if feasible.size == 0:
        return int(depot)

    # Distances
    d_cur = dist[current_node, feasible]
    d_depot = dist[feasible, depot]
    d_depot_to_curr = dist[depot, current_node]

    # Clarke-Wright savings: saving of visiting i after current instead of going to depot
    # saving = d(depot, current) + d(i, depot) - d(current, i)
    # Higher saving = more beneficial to visit i now
    savings = d_depot_to_curr + d_depot - d_cur

    # Capacity pressure: as rest_capacity shrinks relative to remaining demand,
    # we should prefer to close the route sooner
    remaining_demand = float(demands_arr[unvisited].sum())
    if rest_capacity > 0:
        pressure = max(0.0, min(1.0, 1.0 - rest_capacity / max(remaining_demand, 1.0)))
    else:
        pressure = 1.0

    # 1-step lookahead: estimate completion cost after visiting candidate i
    n_after = feasible.size - 1
    if n_after > 0:
        sub = dist[np.ix_(feasible, feasible)]
        mean_edge = float(np.mean(sub)) if sub.size else 0.0
    else:
        mean_edge = 0.0

    cap_after = rest_capacity - demands_arr[feasible]
    mean_demand = float(demands_arr[feasible].mean()) if feasible.size else 1.0
    if mean_demand > 0:
        max_more = np.floor(cap_after / mean_demand).astype(int)
    else:
        max_more = n_after
    max_more = np.clip(max_more, 0, n_after)

    # Estimated continuation cost if we keep going after i
    cont_cost = max_more * mean_edge + d_depot
    ret_cost = d_depot
    completion = np.minimum(cont_cost, ret_cost)

    # Normalize savings and distances for blending
    max_dist = np.max(d_cur) if np.max(d_cur) > 0 else 1.0
    max_saving = np.max(savings) if np.max(savings) > 0 else 1.0

    # Weighted score: primarily savings (minimize negative savings), secondarily distance
    # With capacity pressure, amplify the depot-return tendency
    alpha = 0.5 + 0.3 * pressure  # 0.5 .. 0.8: weight for savings
    beta = 1.0 - alpha            # 0.2 .. 0.5: weight for distance
    
    # Score = beta * (normalized distance) - alpha * (normalized savings) + pressure * completion
    # Lower score = better choice
    norm_dist = d_cur / max_dist
    norm_saving = savings / max_saving
    norm_completion = completion / max_dist if max_dist > 0 else 0.0
    
    scores = beta * norm_dist - alpha * norm_saving + 0.3 * pressure * norm_completion

    idx = int(np.argmin(scores))
    return int(feasible[idx])
# EVOLVE-BLOCK-END
