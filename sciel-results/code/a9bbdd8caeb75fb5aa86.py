"""Guided local search penalty rule for the TSP with hard edge invalidation."""
# EVOLVE-BLOCK-START
import time
import numpy as np

_start_time = time.perf_counter()
_DEADLINE = _start_time + 325.0  # Safety margin against 330s kill


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Perturbation step for Guided Local Search.
    
    Strategy:
    - 'Hard' GLS: Instead of penalizing edges, we completely invalidate 
      (set to a very large value) the edges currently in the local_opt_tour.
    - This forces the 2-opt solver to find a tour that shares NO edges 
      with the previous local optimum (or very few, if 2-opt is not perfect 
      at removing them all, but it will be heavily biased against them).
    - This is a much stronger perturbation than additive penalties, 
      ensuring we jump to a completely different region of the search space.
    - We use a large finite number instead of inf to avoid numerical issues 
      in 2-opt comparisons (e.g., inf - inf = nan).
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

    # Determine a large penalty value.
    # It should be larger than any possible tour length difference to ensure
    # 2-opt avoids these edges if possible.
    # Max possible distance in unit square is sqrt(2) ~ 1.414.
    # Max tour length is 100 * 1.414 = 141.4.
    # Setting to 10000 ensures it's effectively infinite for 2-opt decisions.
    BIG_M = 10000.0

    # Set the distance of all edges in the current tour to BIG_M
    for i in range(n):
        u = local_opt_tour[i]
        v = local_opt_tour[(i + 1) % n]
        D[u, v] = BIG_M
        D[v, u] = BIG_M

    # Ensure finiteness (BIG_M is finite)
    # No need to check isfinite if we just assigned a finite constant, 
    # but it's good practice.
    if not np.all(np.isfinite(D)):
        return edge_distance.copy()

    return D
# EVOLVE-BLOCK-END
