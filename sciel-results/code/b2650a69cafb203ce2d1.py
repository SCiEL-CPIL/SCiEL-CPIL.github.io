"""Capacitated vehicle routing heuristic: regret-2 with savings look-ahead."""
# EVOLVE-BLOCK-START
import time
import numpy as np

_DEADLINE = None
_START_TIME = None


def _ensure_deadline():
    global _DEADLINE, _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
        _DEADLINE = _START_TIME + 140.0  # safety margin before 150s kill


def _time_left():
    _ensure_deadline()
    return _DEADLINE - time.time()


def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    _ensure_deadline()
    if _time_left() < 0.5:
        # Safety: return depot or nearest feasible if any
        if len(unvisited_nodes) == 0:
            return int(depot)
        feasible = [n for n in unvisited_nodes if demands[n] <= rest_capacity]
        if not feasible:
            return int(depot)
        return int(min(feasible, key=lambda n: distance_matrix[current_node, n]))

    unvisited = list(unvisited_nodes)
    if not unvisited:
        return int(depot)

    dm = distance_matrix
    cur = int(current_node)

    # Feasibility filter by capacity
    feasible = [n for n in unvisited if demands[n] <= rest_capacity]
    if not feasible:
        return int(depot)

    # If only one feasible, take it
    if len(feasible) == 1:
        return int(feasible[0])

    # Compute regret-2 score: for each candidate, the gap between best and 2nd-best
    # "alternative" interpretation: if we DON'T visit node i now, how much extra
    # will it cost? Approximate by: distance from current to i, plus the cost of
    # the best remaining connection that would replace it.
    #
    # Regret-2 standard: for node i, regret = (2nd nearest feasible distance from
    # current) - (nearest feasible distance from current). But here "nearest" is
    # always from current, so all nodes share the same "nearest" (current).
    #
    # Better regret: consider the opportunity cost. For each candidate i, compute
    # how "expensive" it would be to skip i and come back later. Approximate:
    #   cost_if_skip = min over j in feasible, j != i of (dm[cur, j] + dm[j, i])
    #   cost_if_take = dm[cur, i]
    #   regret = cost_if_skip - cost_if_take  (higher = more urgent to take now)
    #
    # This is O(n^2) per step but n <= 50, so fine.
    #
    # Also incorporate savings: prefer nodes that have high savings relative to
    # depot (i.e., visiting them in sequence saves distance vs. two separate routes).
    # savings(i) = dm[depot, i] + dm[i, cur] - dm[depot, cur] ... but cur changes.
    # Simpler: use dm[depot, i] as a proxy for "how far from depot" — nodes far
    # from depot benefit more from being grouped.

    n = len(feasible)
    # Precompute distances from current to all feasible
    dist_from_cur = np.array([dm[cur, i] for i in feasible])

    # Compute regret for each feasible node
    regrets = np.zeros(n)
    for idx, i in enumerate(feasible):
        if n == 1:
            regrets[idx] = 0.0
            continue
        # Cost if we skip i: go to some j first, then to i
        # min over j != i of dm[cur, j] + dm[j, i]
        min_skip_cost = np.inf
        for jdx, j in enumerate(feasible):
            if j == i:
                continue
            c = dist_from_cur[jdx] + dm[j, i]
            if c < min_skip_cost:
                min_skip_cost = c
        cost_take = dist_from_cur[idx]
        regrets[idx] = min_skip_cost - cost_take

    # Savings component: prefer nodes with higher dm[depot, i] (far from depot)
    # because they benefit more from being on the same route.
    # Also consider the angle/position relative to current direction.
    # Use dm[depot, i] as a simple proxy, normalized.
    savings_scores = np.array([dm[int(depot), i] for i in feasible])

    # Combine: regret is primary, savings is secondary tie-breaker
    # Scale savings to be comparable to regret
    if regrets.max() > 0:
        regret_norm = regrets / regrets.max()
    else:
        regret_norm = regrets
    if savings_scores.max() > 0:
        sav_norm = savings_scores / savings_scores.max()
    else:
        sav_norm = savings_scores

    # Weighted score: higher regret = more urgent, higher savings = better to group
    combined = regret_norm + 0.3 * sav_norm

    # Also add a small nearest-neighbor bias to avoid pathological long jumps
    if dist_from_cur.max() > 0:
        dist_norm = dist_from_cur / dist_from_cur.max()
    else:
        dist_norm = dist_from_cur
    combined = combined - 0.2 * dist_norm

    best_idx = int(np.argmax(combined))
    return int(feasible[best_idx])
# EVOLVE-BLOCK-END
