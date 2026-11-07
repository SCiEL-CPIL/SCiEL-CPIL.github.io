"""Guided local search penalty rule for the TSP – adaptive multi-level penalty."""
import numpy as np
import time

_START_TIME = time.time()
_TIME_LIMIT = 315.0  # Safety margin before 330s kill


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Guided local search penalty rule for TSP.

    Strategy:
    - Infer the current round from edge_n_used (max value).
    - Apply an additive penalty to edges in the current local-opt tour.
    - The penalty magnitude scales with usage count and round number,
      normalized by the tour's average edge length so it is problem-relative.
    - Also apply a smaller global penalty to frequently-used edges not in
      the current tour to help escape persistent local optima.
    - Ensure the returned matrix is symmetric, finite, and has zero diagonal.
    """
    # Time guard
    elapsed = time.time() - _START_TIME
    if elapsed > _TIME_LIMIT:
        return edge_distance.copy()

    n = edge_distance.shape[0]
    tour = local_opt_tour

    if n < 2 or len(tour) < 2:
        return edge_distance.copy()

    # --- Extract tour edges and compute scale ---
    tour_edges = []
    tour_lens = []
    for i in range(len(tour)):
        u = tour[i]
        v = tour[(i + 1) % len(tour)]
        tour_edges.append((u, v))
        tour_lens.append(edge_distance[u, v])

    avg_len = float(np.mean(tour_lens)) if tour_lens else 1.0
    if avg_len < 1e-9:
        avg_len = 1.0

    # Infer round number from max usage
    max_used = int(edge_n_used.max()) if edge_n_used.size > 0 else 0
    round_num = max_used + 1  # 1-based round estimate

    # --- Build penalty matrix ---
    penalty = np.zeros((n, n), dtype=np.float64)

    # Penalty for edges in the current tour:
    #   base coefficient grows gently with round number,
    #   scaled by usage so previously-penalised edges get more push.
    base_coeff = 0.3 * avg_len * (1.0 + 0.1 * np.log1p(round_num))

    for u, v in tour_edges:
        usage = float(edge_n_used[u, v])
        # Normalised usage in [0, ~1]
        norm_usage = usage / (1.0 + max_used)
        # Edge-specific penalty: base * (0.5 + 0.8 * norm_usage)
        # The 0.5 floor ensures even first-use edges are penalised;
        # the 0.8 factor amplifies penalty for heavily-used edges.
        ep = base_coeff * (0.5 + 0.8 * norm_usage)
        penalty[u, v] += ep
        penalty[v, u] += ep

    # Global penalty for frequently-used edges NOT in the current tour.
    # This discourages the solver from re-forming tours that share many
    # high-usage edges with past local optima.
    global_coeff = 0.05 * avg_len
    # Normalised usage matrix
    norm_usage_mat = edge_n_used.astype(np.float64) / (1.0 + max_used)

    # Mask: True where NOT a current-tour edge (and not diagonal)
    mask = np.ones((n, n), dtype=bool)
    for u, v in tour_edges:
        mask[u, v] = False
        mask[v, u] = False
    np.fill_diagonal(mask, False)

    # Apply global penalty only where mask is True and usage > 0
    global_pen = np.where(mask & (norm_usage_mat > 0),
                          global_coeff * norm_usage_mat, 0.0)
    penalty += global_pen

    # --- Combine ---
    result = edge_distance + penalty

    # Ensure symmetry
    result = (result + result.T) * 0.5
    # Ensure finite
    result = np.nan_to_num(result, nan=0.0,
                           posinf=np.finfo(np.float64).max / 10,
                           neginf=0.0)
    # Zero diagonal
    np.fill_diagonal(result, 0.0)

    return result
