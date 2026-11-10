"""Route-level look-ahead heuristic for Capacitated Vehicle Routing Problem.

Combines the current program's route-completion simulation with a capacity-aware
savings boost: nodes with high demand that fit now are 'hard to place' later
since capacity only decreases. We add a demand-consolidation bonus to the
predicted completion cost to encourage packing large nodes into the current route.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_TIMEOUT = 115.0  # Safety margin before 120s limit
_DEADLINE = _start_time + _TIMEOUT


def _complete_route_nn(start, remaining_cap, unvisited_set, demands, dist, depot):
    """
    Fast nearest-neighbor-with-capacity completion of a route.

    Returns total_extra_distance: sum of leg distances from `start` through
    the greedily-chosen remaining nodes and back to the depot.
    """
    total = 0.0
    cur = start
    cap = remaining_cap
    remaining = list(unvisited_set)

    while remaining:
        feasible = [n for n in remaining if demands[n] <= cap]
        if not feasible:
            break
        best_n = None
        best_d = np.inf
        for n in feasible:
            d = dist[cur, n]
            if d < best_d:
                best_d = d
                best_n = n
        total += best_d
        cap -= demands[best_n]
        cur = best_n
        remaining.remove(best_n)
        if cap <= 0:
            break

    total += dist[cur, depot]
    return total


def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Pick the next customer by minimizing the predicted total length of the
    current route, with a capacity-aware demand-consolidation bonus.

    Returns 0 (depot) if no feasible node or time limit reached.
    """
    if time.time() > _DEADLINE:
        return 0

    if len(unvisited_nodes) == 0:
        return 0

    unvisited = list(unvisited_nodes)
    dist = distance_matrix

    # Filter by capacity
    feasible = [n for n in unvisited if demands[n] <= rest_capacity]
    if not feasible:
        return 0

    best_c = None
    best_score = np.inf

    # Precompute some constants for the demand bonus
    # The idea: if a node has high demand relative to remaining capacity,
    # it's harder to place later. Give a bonus (negative cost) for picking
    # such nodes now. The bonus scales with demand and inversely with
    # remaining capacity pressure.
    # bonus = -lambda * demand[n] / (1 + rest_capacity / 10)
    # This is a soft penalty for not consolidating large demands.
    # We tune lambda to be modest so it doesn't override the completion cost.

    for c in feasible:
        if time.time() > _DEADLINE:
            break
        d_first = dist[current_node, c]
        new_cap = rest_capacity - demands[c]

        # Completion set: all feasible-except-c
        remaining_for_completion = [n for n in feasible if n != c]
        extra = _complete_route_nn(c, new_cap, remaining_for_completion, demands, dist, depot)

        # Base score: predicted total route length from here
        base_score = d_first + extra

        # Demand consolidation bonus:
        # High-demand nodes that fit now should be preferred because they
        # become infeasible later as capacity shrinks.
        # The bonus is proportional to demand and inversely proportional to
        # how much spare capacity we have (more spare = less urgent).
        # Use a gentle penalty: -0.5 * demand[c] * (1.0 / (1.0 + new_cap / 5.0))
        # When new_cap is large, the bonus is small. When new_cap is small,
        # the bonus is larger, encouraging us to take the big node now.
        demand_bonus = -0.5 * demands[c] * (1.0 / (1.0 + new_cap / 5.0))

        score = base_score + demand_bonus

        if score < best_score:
            best_score = score
            best_c = c

    if best_c is None:
        # Time ran out during the loop; fall back to nearest feasible node
        best_c = min(feasible, key=lambda n: dist[current_node, n])

    return int(best_c)
# EVOLVE-BLOCK-END
