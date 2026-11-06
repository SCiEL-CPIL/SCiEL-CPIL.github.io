"""Capacitated vehicle routing – regret-2 selection with time guard."""
# EVOLVE-BLOCK-START
import time
import numpy as np

_DEADLINE = 120.0
_START = time.monotonic()


def _elapsed():
    return time.monotonic() - _START


def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    # Safety: return depot if we're close to the deadline
    if _elapsed() > _DEADLINE - 5.0:
        return int(depot)

    unvisited_nodes = np.asarray(unvisited_nodes, dtype=int)
    if len(unvisited_nodes) == 0:
        return int(depot)

    n = len(unvisited_nodes)
    dmat = np.asarray(distance_matrix)

    # Feasible nodes: demand fits remaining capacity
    feasible_mask = demands[unvisited_nodes] <= rest_capacity
    feasible_nodes = unvisited_nodes[feasible_mask]

    if len(feasible_nodes) == 0:
        return int(depot)

    # Distance from current to all feasible nodes
    dists = dmat[current_node, feasible_nodes]

    # Regret-2: for each feasible node, compute the cost of NOT choosing it now
    # i.e., the difference between the best and second-best alternative
    # We want to pick the node with the highest regret (most unique position)

    # For each feasible node i, regret = min_{j != i, j feasible} dist(j) - dist(i)
    # But we also need to consider the return-to-depot cost for closing the route.

    # Build a score: regret + a small penalty for high demand (to save capacity for harder-to-place nodes)
    # Actually, let's use a cleaner approach:
    # Score = -regret + alpha * demand (we want to pick high regret, low demand)
    # But first compute regret properly.

    # For efficiency with up to 50 nodes, we can do it with a small loop
    # or use numpy tricks.

    if n <= 1:
        return int(feasible_nodes[0])

    # Compute for each feasible node the regret:
    # regret[i] = (second smallest distance among all feasible nodes) - (smallest distance if we skip i)
    # Actually the standard regret-k definition: for node i, regret = sum of (k-th best - 1st best) costs.
    # For regret-2: regret(i) = min_{j != i} cost(j) - cost(i)  ... no that's not right either.

    # Let me use the standard definition:
    # For each candidate node i, the regret is:
    #   r(i) = (2nd best option cost) - (1st best option cost)
    # where "option cost" for node i is dist(current, i) + dist(i, depot) if it were the last,
    # but more generally we use just the immediate distance.

    # Simpler regret-2: 
    # For node i, sort all feasible nodes by distance from current.
    # Regret(i) = dist(2nd closest) - dist(i)  if i is the closest
    # More generally: regret(i) = min_{j in feasible, j != i} dist(j) - dist(i)
    # This can be negative if i is not the closest.

    # Let's compute: for each i, the minimum distance to any other feasible node
    # regret[i] = min_{j != i} dists[j] - dists[i]

    # To compute min_{j != i} dists[j] efficiently:
    # Find the global min and second min of dists
    sorted_idx = np.argsort(dists)
    min1_val = dists[sorted_idx[0]]
    min2_val = dists[sorted_idx[1]] if n > 1 else min1_val
    min1_idx = sorted_idx[0]

    # For each i:
    #   if i == min1_idx: best_other = min2_val
    #   else: best_other = min1_val
    best_other = np.where(
        feasible_nodes == feasible_nodes[min1_idx],
        min2_val,
        min1_val
    )
    regret = best_other - dists

    # Combine regret with a slight preference for lower demand (to keep capacity flexible)
    # and distance to depot (prefer nodes that are not too far from depot to close)
    demand_factor = 0.01 * demands[feasible_nodes]
    depot_dist = dmat[feasible_nodes, depot] * 0.05

    score = regret - demand_factor - depot_dist
    best_idx = np.argmax(score)
    return int(feasible_nodes[best_idx])
# EVOLVE-BLOCK-END
