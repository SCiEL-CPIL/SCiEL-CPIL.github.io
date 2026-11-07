"""Guided local search penalty rule for the TSP – combined adaptive penalty."""
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
    - Use a hybrid scaling for usage: strong initial penalty that grows
      with usage but saturates to avoid unbounded growth.
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

    # Work in float64
    D = np.asarray(edge_distance, dtype=np.float64)
    U = np.asarray(edge_n_used, dtype=np.float64)

    # --- Extract tour edges and compute scale ---
    tour_edges = []
    tour_lens = []
    for i in range(len(tour)):
        u = tour[i]
        v = tour[(i + 1) % len(tour)]
        tour_edges.append((u, v))
        tour_lens.append(D[u, v])

    avg_len = float(np.mean(tour_lens)) if tour_lens else 1.0
    if avg_len < 1e-9:
        avg_len = 1.0

    # Infer round number from max usage
    max_used = int(U.max()) if U.size > 0 else 0
    round_num = max(1, max_used + 1)  # 1-based round estimate

    # --- Build penalty matrix ---
    penalty = np.zeros((n, n), dtype=np.float64)

    # Penalty for edges in the current tour:
    #   Use a hybrid scaling that starts strong and grows with usage.
    #   The base coefficient grows gently with round number.
    base_coeff = 0.25 * avg_len * (1.0 + 0.12 * np.log1p(round_num))

    for u, v in tour_edges:
        usage = float(U[u, v])
        # Hybrid scaling: combine linear growth with soft saturation
        # For low usage (0-2), behaves roughly linearly
        # For high usage, saturates but still grows
        hybrid_usage = usage + 0.5 * usage / (1.0 + 0.5 * usage)
        # Edge-specific penalty with a floor to ensure first-use edges are penalized
        scale_factor = max(1.0, float(round_num))
        ep = base_coeff * (0.5 + 1.0 * hybrid_usage / scale_factor)
        penalty[u, v] += ep
        penalty[v, u] += ep

    # Global penalty for frequently-used edges NOT in the current tour.
    # This discourages the solver from re-forming tours that share many
    # high-usage edges with past local optima.
    global_coeff = 0.03 * avg_len
    # Use soft saturation for global penalty to keep it bounded
    soft_usage_mat = U / (1.0 + U)

    # Mask: True where NOT a current-tour edge (and not diagonal)
    mask = np.ones((n, n), dtype=bool)
    for u, v in tour_edges:
        mask[u, v] = False
        mask[v, u] = False
    np.fill_diagonal(mask, False)

    # Apply global penalty only where mask is True and usage > 0
    global_pen = np.where(mask & (soft_usage_mat > 0),
                          global_coeff * soft_usage_mat, 0.0)
    penalty += global_pen

    # --- Combine ---
    result = D + penalty

    # Ensure symmetry
    result = (result + result.T) * 0.5
    # Ensure finite
    result = np.nan_to_num(result, nan=0.0,
                           posinf=np.finfo(np.float64).max / 10,
                           neginf=0.0)
    # Zero diagonal
    np.fill_diagonal(result, 0.0)

    return result
