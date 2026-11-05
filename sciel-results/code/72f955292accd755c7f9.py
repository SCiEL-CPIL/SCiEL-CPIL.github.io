"""Constructive TSP heuristic: return-aware (closure) nearest neighbour with adaptive alpha.

MODEL-AND-OPTIMIZE:
    We model the cost contribution of picking candidate c (from current node,
    toward the fixed destination that the tour must close to) as a linear
    combination of the two edges that are directly affected by this choice:
        score(c) = dist(current, c)  +  alpha * dist(c, destination)
    - The first term is the classic nearest-neighbour (NN) edge cost.
    - The second term is a proxy for the eventual *closing* cost: the tour must
      return to `destination`, so a candidate far from it risks a long final
      edge. Pure NN ignores this and systematically "drifts" away from the
      start, producing a long closing edge on random Euclidean instances.
    `alpha` is the fitted weight on the closure term. We use an adaptive
    schedule: higher alpha early (when the tour is short and drift is most
    dangerous) and lower alpha late (when we are closer to the destination
    and pure NN is more reliable). This reduces over-penalization near the
    end while still preventing early drift.
"""
# EVOLVE-BLOCK-START
import time
import numpy as np

# Explicit deadline guard: the evaluator kills after 150 s wall-clock.
# Each call is O(|unvisited|) vectorised NumPy work (microseconds for n=50),
# so this guard is only a safety net; it does not gate the search.
_RUN_START = time.perf_counter()
_DEADLINE_S = 110.0  # safety margin well under the 150 s kill switch


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    # Safety: if (unexpectedly) no candidates remain, close to the destination.
    if unvisited_nodes is None or len(unvisited_nodes) == 0:
        return int(destination_node)

    # Deadline guard: if we are dangerously close to the wall-clock limit,
    # fall back to the cheapest deterministic choice (pure NN) immediately.
    if (time.perf_counter() - _RUN_START) > _DEADLINE_S:
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])

    # Adaptive alpha: start high (1.5) to prevent drift, decay to low (0.2)
    # as we get close to the destination. The fraction of unvisited nodes
    # remaining is a proxy for how much of the tour is left to complete.
    n_unvisited = len(unvisited_nodes)
    # For 50 cities, we start with 49 unvisited and end with 1.
    # fraction_remaining = n_unvisited / 49, so it goes from ~1.0 to ~0.02
    fraction_remaining = n_unvisited / 49.0
    # Linear interpolation: alpha = 0.2 + (1.5 - 0.2) * fraction_remaining
    # When fraction_remaining=1.0 (start), alpha=1.5
    # When fraction_remaining=0.02 (end), alpha=0.2 + 1.3*0.02 = 0.226
    alpha = 0.2 + 1.3 * fraction_remaining

    # Vectorised return-aware nearest-neighbour score.
    d_current = distance_matrix[current_node, unvisited_nodes]   # edge into candidate
    d_dest    = distance_matrix[destination_node, unvisited_nodes]  # proxy closing cost
    scores = d_current + alpha * d_dest
    return int(unvisited_nodes[np.argmin(scores)])
# EVOLVE-BLOCK-END
