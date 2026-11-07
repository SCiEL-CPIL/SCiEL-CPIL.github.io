"""Guided local search penalty rule for the TSP – multiplicative context-aware penalty."""
import numpy as np
import time

_START_TIME = time.time()
_TIME_LIMIT = 315.0  # Safety margin before 330s kill


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Multiplicative context-aware penalty for guided local search on TSP.

    Key insight from observations: multiplicative penalties that target
    only current tour edges preserve global geometry for 2-opt moves while
    specifically breaking long, frequently-used 'stuck' edges.

    Strategy:
    - For each edge in the current local-opt tour, apply a multiplicative
      penalty factor that grows with both the edge's usage count and its
      relative length compared to the tour average.
    - Edges that are both long and frequently used get the strongest penalty,
      as they are the most likely to be 'stuck' edges preventing escape from
      local optima.
    - A small additive component ensures even short, frequently-used edges
      are penalized to some degree.
    - No global penalties are applied, preserving the distance structure
      for edges outside the current tour.
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
    D = np.asarray(edge_distance, dtype=np.float64).copy()
    U = np.asarray(edge_n_used, dtype=np.float64)

    # --- Extract tour edges and compute statistics ---
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

    # Infer round number from max usage count
    max_used = int(U.max()) if U.size > 0 else 0
    round_num = max_used + 1  # 1-based round estimate

    # --- Apply refined multiplicative penalty to tour edges ---
    # Adaptive base multiplier: gentler early, stronger later
    # Use a smooth S-curve-like growth that accelerates in middle rounds
    if round_num <= 1:
        base_mult = 1.0
    elif round_num <= 5:
        base_mult = 1.0 + 0.05 * (round_num - 1)
    elif round_num <= 10:
        base_mult = 1.2 + 0.08 * (round_num - 5)
    else:
        base_mult = 1.6 + 0.03 * (round_num - 10)

    for u, v in tour_edges:
        d_uv = D[u, v]
        usage = float(U[u, v])

        # Relative length factor: longer edges get more penalty
        # Use a smoother mapping with tighter clamp
        rel_len = d_uv / avg_len if avg_len > 0 else 1.0
        rel_len = np.clip(rel_len, 0.7, 2.5)

        # Combined "stuckness" score: high usage AND long relative to avg
        # Use a smooth product-like combination via log-sum-exp inspired approach
        # stuck_score = exp(a*log(usage+1) + b*log(rel_len+0.1)) / normalizer
        # Simplified: geometric mean of normalized factors
        usage_norm = np.log1p(usage) / np.log1p(max(1.0, float(U.max()) + 1)) if U.size > 0 else 0.0
        rel_norm = (rel_len - 0.7) / (2.5 - 0.7)  # normalize to [0,1]
        
        # Combined stuckness: weighted combination favoring edges that are both
        # frequently used and longer than average
        stuckness = 0.4 * usage_norm + 0.6 * rel_norm
        stuckness = np.clip(stuckness, 0.0, 1.0)

        # Multiplicative penalty: base * (1 + strength * stuckness)
        # Strength grows with round to progressively break more edges
        strength = 0.8 + 0.4 * np.log1p(round_num)
        mult_factor = base_mult * (1.0 + strength * stuckness)

        # Small additive component scaled by usage for short edges
        add_component = 0.015 * avg_len * np.log1p(usage) * (1.0 - 0.5 * rel_norm)

        new_d = d_uv * mult_factor + add_component

        D[u, v] = new_d
        D[v, u] = new_d

    # Ensure symmetry
    D = (D + D.T) * 0.5
    # Ensure finite
    D = np.nan_to_num(D, nan=0.0,
                      posinf=np.finfo(np.float64).max / 10,
                      neginf=0.0)
    # Zero diagonal
    np.fill_diagonal(D, 0.0)

    return D
