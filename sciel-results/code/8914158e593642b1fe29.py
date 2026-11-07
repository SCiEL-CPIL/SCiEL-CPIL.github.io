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

    # If no edges have been used yet, return original
    max_usage = U.max()
    if max_usage == 0:
        return D.copy()

    # Extract tour edges
    tour_edges = np.zeros((n, n), dtype=bool)
    tour_lens = []
    for i in range(n):
        a = local_opt_tour[i]
        b = local_opt_tour[(i + 1) % n]
        tour_edges[a, b] = True
        tour_edges[b, a] = True
        tour_lens.append(D[a, b])

    avg_edge_len = np.mean(tour_lens) if tour_lens else 1.0
    if avg_edge_len < 1e-9:
        avg_edge_len = 1.0

    # Infer round number from max usage
    round_num = int(max_usage) + 1

    # Build penalty matrix
    penalty = np.zeros((n, n), dtype=np.float64)

    # Penalty for edges in the current tour:
    #   base coefficient grows gently with round number,
    #   scaled by usage via soft saturation u/(1+u).
    # This combines the multiplicative scaling idea from the current program
    # with the additive penalty structure from the inspirations.
    base_coeff = 0.2 * avg_edge_len * (1.0 + 0.08 * np.log1p(round_num))

    for i in range(n):
        a = local_opt_tour[i]
        b = local_opt_tour[(i + 1) % n]
        usage = float(U[a, b])
        # Soft saturation: u/(1+u) is in [0, 1), monotone, bounded
        soft_usage = usage / (1.0 + usage)
        # Edge-specific penalty: base * (0.3 + 0.6 * soft_usage)
        # Reduced floor and amplification for gentler perturbation
        ep = base_coeff * (0.3 + 0.6 * soft_usage)
        penalty[a, b] += ep
        penalty[b, a] += ep

    # Global penalty for frequently-used edges NOT in the current tour.
    # This discourages the solver from re-forming tours that share many
    # high-usage edges with past local optima.
    global_coeff = 0.03 * avg_edge_len
    # Soft-saturated usage matrix
    soft_usage_mat = U / (1.0 + U)

    # Mask: True where NOT a current-tour edge (and not diagonal)
    mask = ~tour_edges
    np.fill_diagonal(mask, False)

    # Apply global penalty only where mask is True and usage > 0
    global_pen = np.where(mask & (soft_usage_mat > 0),
                          global_coeff * soft_usage_mat, 0.0)
    penalty += global_pen

    # --- Combine: additive penalty on top of true distances ---
    # Scale penalty slightly to avoid over-perturbation
    result = D + 0.8 * penalty

    # Ensure symmetry
    result = (result + result.T) * 0.5
    # Ensure finite
    result = np.nan_to_num(result, nan=0.0,
                           posinf=np.finfo(np.float64).max / 10,
                           neginf=0.0)
    # Zero diagonal
    np.fill_diagonal(result, 0.0)

    return result
# EVOLVE-BLOCK-END
