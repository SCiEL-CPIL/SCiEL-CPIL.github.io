"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_GLOBAL_START_TIME = time.time()
_WALL_CLOCK_LIMIT = 320.0  # Safety margin before 330s kill


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    # Deadline guard: if we're close to the limit, return original distances
    if time.time() - _GLOBAL_START_TIME > _WALL_CLOCK_LIMIT:
        return edge_distance.copy()

    n = edge_distance.shape[0]
    if n < 2:
        return edge_distance.copy()

    D = np.asarray(edge_distance, dtype=np.float64)
    U = np.asarray(edge_n_used, dtype=np.float64)

    max_usage = U.max()
    if max_usage == 0:
        return D.copy()

    # Build tour edge mask
    tour_edge_mask = np.zeros((n, n), dtype=bool)
    for i in range(n):
        a = local_opt_tour[i]
        b = local_opt_tour[(i + 1) % n]
        tour_edge_mask[a, b] = True
        tour_edge_mask[b, a] = True

    # Get usage counts for tour edges
    tour_usage = U[tour_edge_mask]

    # Compute penalties with a smooth, saturating function
    # alpha: overall penalty strength
    # beta: controls how quickly penalty saturates
    # gamma: additional scaling for very high usage
    alpha = 3.0
    beta = 0.8
    gamma = 0.3

    # Normalize usage to [0, 1]
    norm_usage = tour_usage / max_usage

    # Soft saturation: 1 - exp(-beta * norm_usage) gives a smooth curve
    # that starts steep and flattens out
    base_penalty = 1.0 + alpha * (1.0 - np.exp(-beta * norm_usage))

    # Additional penalty for edges used many times (non-linear boost)
    high_usage_boost = 1.0 + gamma * (norm_usage ** 2)

    penalty_factor = base_penalty * high_usage_boost

    # Apply penalty only to tour edges
    result = D.copy()
    result[tour_edge_mask] *= penalty_factor

    # Ensure diagonal stays zero
    np.fill_diagonal(result, 0.0)

    # Ensure symmetry
    result = (result + result.T) / 2.0

    return result
# EVOLVE-BLOCK-END
