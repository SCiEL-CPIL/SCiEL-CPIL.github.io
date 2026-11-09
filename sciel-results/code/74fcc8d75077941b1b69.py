"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    GLS penalty: penalize edges in the current local optimum tour
    based on their historical usage frequency.
    """
    t0 = time.time()
    deadline = t0 + 280.0  # safety margin before 300s kill

    n = len(local_opt_tour)
    if n <= 1:
        return edge_distance.copy()

    result = edge_distance.copy()

    # Scale penalty relative to typical edge distances
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    med_dist = np.median(edge_distance[mask])
    if med_dist <= 0:
        med_dist = 1.0

    # Penalty per unit of usage count
    penalty_coeff = 0.15 * med_dist

    # Only penalize edges that are in the current local optimum tour
    for i in range(n):
        u = local_opt_tour[i]
        v = local_opt_tour[(i + 1) % n]
        result[u, v] += penalty_coeff * edge_n_used[u, v]
        result[v, u] += penalty_coeff * edge_n_used[v, u]

    # Check time
    if time.time() > deadline:
        return result

    # Additional refinement: apply a smooth penalty that encourages diversification
    # Use soft-max style weighting based on edge usage
    # This creates a smoother landscape for the local search
    if time.time() < deadline - 10.0:
        # Normalize usage counts
        max_used = edge_n_used.max()
        if max_used > 0:
            normalized_used = edge_n_used / (1.0 + max_used)
            # Apply a quadratic penalty for frequently used edges
            quad_penalty = 0.05 * med_dist * normalized_used ** 2
            # Only apply to edges in the current tour
            for i in range(n):
                u = local_opt_tour[i]
                v = local_opt_tour[(i + 1) % n]
                result[u, v] += quad_penalty[u, v]
                result[v, u] += quad_penalty[v, u]

    return result
# EVOLVE-BLOCK-END
