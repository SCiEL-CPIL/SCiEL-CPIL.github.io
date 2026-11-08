"""Guided local search penalty rule for the TSP with conditional edge penalties."""
# EVOLVE-BLOCK-START
import time
import numpy as np

_start_time = time.perf_counter()
_DEADLINE = _start_time + 325.0  # Safety margin against 330s kill


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Perturbation step for Guided Local Search.
    
    Strategy: 
    - Only penalize edges that are currently in the local_opt_tour.
    - Penalty magnitude is proportional to sqrt(edge_n_used[i,j]).
    - This creates a strong, localized repulsion from the specific 
      local optimum structure, rather than a global distortion.
    - Edges not in the tour remain at their true distance, preserving 
      the metric structure for potential new connections.
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

    # Extract the edges in the current tour
    # Tour is a closed loop: 0->1->...->n-1->0
    tour_edges = []
    for i in range(n):
        u = local_opt_tour[i]
        v = local_opt_tour[(i + 1) % n]
        tour_edges.append((u, v))

    # Apply penalties to these specific edges
    # We use a coefficient that grows with usage but saturates to prevent 
    # infinite distances. sqrt usage is a common heuristic in GLS variants.
    # Base penalty scale: we want the penalty to be significant but not 
    # overwhelming. 1.0 is a reasonable start, but we can tune it.
    # Let's use a penalty that makes the edge 50% more expensive if used once, 
    # 70% if used twice, etc. via a multiplicative factor.
    
    # Alternative: Additive penalty based on usage.
    # If edge used k times, add k * alpha to distance.
    # Let's try a multiplicative penalty that is stronger for frequently used edges.
    # penalty_factor = 1.0 + beta * sqrt(count)
    
    beta = 0.5  # Tunable parameter
    
    for u, v in tour_edges:
        count = edge_n_used[u, v]
        if count > 0:
            # Multiplicative penalty
            factor = 1.0 + beta * np.sqrt(count)
            D[u, v] *= factor
            D[v, u] *= factor

    # Ensure symmetry and finiteness
    # The multiplication preserves symmetry if input was symmetric.
    # Check for any non-finite values (shouldn't happen with finite input and positive factors)
    if not np.all(np.isfinite(D)):
        # Fallback to true distances if something went wrong
        return edge_distance.copy()

    return D
# EVOLVE-BLOCK-END
