"""Guided local search penalty rule for TSP using 2-opt move regret model."""
# EVOLVE-BLOCK-START
import time
import numpy as np

_start_time = time.perf_counter()
_DEADLINE = _start_time + 325.0  # Safety margin against 330s kill


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Perturbation step for Guided Local Search.
    
    Strategy:
    - 2-opt Move Regret Model: For each edge in the current tour, calculate the
      minimal length increase (regret) required to remove it via a single 2-opt swap.
      Edges with low regret are 'fragile' (easy to remove) and have been part of
      the tour structure persistently. Edges with high regret are 'robust'.
    - We penalize edges based on their regret: edges with LOWER regret get HIGHER
      penalty, forcing the 2-opt search to break these fragile edges first.
      This creates a targeted disruption of the current basin's weak points.
    - Additionally, we apply a small global usage-based penalty to discourage
      all historically frequent edges, but the regret-based penalty dominates.
    - The penalty is scaled by the average edge length to be instance-invariant.
    """
    # Check deadline: if we are too close to the limit, return a safe, 
    # minimally perturbed matrix to avoid timeout.
    if time.perf_counter() > _DEADLINE:
        return edge_distance.copy()

    n = len(local_opt_tour)
    if n < 2:
        return edge_distance.copy()

    # Create a copy of the true distance matrix
    D = edge_distance.copy()

    # Extract the tour edges
    tour = np.array(local_opt_tour)
    tour_len = 0.0
    for i in range(n):
        u = tour[i]
        v = tour[(i + 1) % n]
        tour_len += edge_distance[u, v]
    
    avg_edge_len = tour_len / n

    # Calculate regret for each edge in the current tour
    # For edge (u, v) at position i in the tour, a 2-opt swap removes edges
    # (tour[i], tour[i+1]) and (tour[j], tour[j+1]) and adds (tour[i], tour[j]) 
    # and (tour[i+1], tour[j+1]).
    # To remove edge (tour[i], tour[i+1]), we need to find the best j such that
    # the swap improves the tour. The regret is the minimum cost increase.
    # Actually, the regret for removing edge e_i is the minimum over all j of:
    #   (d(tour[i], tour[j]) + d(tour[i+1], tour[j+1]) - d(tour[i], tour[i+1]) - d(tour[j], tour[j+1]))
    # where the result is positive (i.e., the swap would increase length).
    # If all swaps decrease length, the edge is not in a local optimum, but we assume
    # local_opt_tour is 2-opt optimal, so all such deltas should be >= 0.
    # The regret is the minimum of these positive deltas.

    # For efficiency, we compute this for each edge in the tour.
    # We only consider j != i and j != i-1 (mod n) to avoid trivial swaps.
    
    regrets = np.zeros(n)
    for i in range(n):
        u = tour[i]
        v = tour[(i + 1) % n]
        min_regret = np.inf
        for j in range(n):
            if j == i or j == (i - 1) % n:
                continue
            a = tour[j]
            b = tour[(j + 1) % n]
            # Swap removes (u,v) and (a,b), adds (u,a) and (v,b)
            # Or removes (u,v) and (a,b), adds (u,b) and (v,a) -- 2-opt has two variants
            # Standard 2-opt: reverse the segment between i and j.
            # This removes (u, v) and (a, b) and adds (u, a) and (v, b).
            delta1 = edge_distance[u, a] + edge_distance[v, b] - edge_distance[u, v] - edge_distance[a, b]
            # The other variant removes (u,v) and (a,b) and adds (u,b) and (v,a)
            # This corresponds to reversing the segment in the other direction, 
            # which is equivalent to swapping i and j in a different way.
            # Actually, for a given pair of edges (i, j), there's only one 2-opt move
            # that removes both. Let's stick with the standard definition.
            if delta1 < min_regret:
                min_regret = delta1
        
        # If no valid j found (n < 3), set regret to a large value
        if np.isinf(min_regret):
            min_regret = 0.0
        
        regrets[i] = min_regret

    # Now, we want to penalize edges with LOW regret (easy to remove) more heavily.
    # Normalize regrets to [0, 1] range
    if np.max(regrets) > 0:
        normalized_regrets = regrets / np.max(regrets)
    else:
        normalized_regrets = np.zeros(n)

    # Penalty for each edge in the tour: higher penalty for lower regret
    # penalty = alpha * avg_edge_len * (1 - normalized_regret)
    alpha = 2.0  # Strong penalty factor
    for i in range(n):
        u = tour[i]
        v = tour[(i + 1) % n]
        penalty = alpha * avg_edge_len * (1.0 - normalized_regrets[i])
        D[u, v] += penalty
        D[v, u] += penalty

    # Additionally, apply a small global usage-based penalty
    max_count = np.max(edge_n_used)
    if max_count > 0:
        normalized_counts = edge_n_used / max_count
        # Small global penalty, 10% of the regret-based penalty scale
        global_penalty = 0.1 * alpha * avg_edge_len * (normalized_counts ** 2)
        D += global_penalty

    # Ensure symmetry
    D = (D + D.T) / 2.0

    # Ensure finiteness
    if not np.all(np.isfinite(D)):
        return edge_distance.copy()

    return D
# EVOLVE-BLOCK-END
