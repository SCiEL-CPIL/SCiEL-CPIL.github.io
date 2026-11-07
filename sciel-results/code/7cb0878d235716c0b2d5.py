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
    result = edge_distance.copy()

    # Build a mask of edges in the current tour
    tour_edge_mask = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        a = local_opt_tour[i]
        b = local_opt_tour[(i + 1) % n]
        tour_edge_mask[a, b] = 1.0
        tour_edge_mask[b, a] = 1.0

    # Get the usage counts for tour edges
    tour_edge_usage = edge_n_used * tour_edge_mask

    # Compute per-edge penalty: stronger for more-used edges
    # Use a scaling that makes heavily-used edges get much larger penalties
    # max_usage helps normalize so the penalty is meaningful
    max_usage = edge_n_used.max()
    if max_usage == 0:
        max_usage = 1.0

    # Penalty factor: 1 + alpha * (usage / max_usage)^beta
    # alpha controls overall strength, beta controls nonlinearity
    alpha = 2.0
    beta = 1.5
    normalized_usage = tour_edge_usage / max_usage
    penalty_factor = 1.0 + alpha * (normalized_usage ** beta)

    # Apply penalty only to tour edges
    result = result * penalty_factor

    # Ensure the diagonal stays zero (no self-loops)
    np.fill_diagonal(result, 0.0)

    # Ensure symmetry (should already be, but just in case)
    result = (result + result.T) / 2.0

    return result
# EVOLVE-BLOCK-END
