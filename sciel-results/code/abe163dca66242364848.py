"""Guided local search penalty rule for the TSP – linear adaptive penalty."""
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
    - The penalty magnitude scales linearly with the raw usage count,
      providing progressively stronger pressure to escape local optima
      in later iterations.
    - The base penalty is normalized by the tour's average edge length.
    - Also apply a smaller global penalty to frequently-used edges not in
      the current tour, also scaled linearly by usage.
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
    round_num = max_used + 1  # 1-based round estimate

    # --- Build penalty matrix ---
    penalty = np.zeros((n, n), dtype=np.float64)

    # Penalty for edges in the current tour:
    #   base coefficient is kept relatively constant to let the linear
    #   usage term drive the growth in penalty strength.
    #   We use a slightly smaller base coefficient than before to account
    #   for the larger dynamic range of the linear usage term.
    base_coeff = 0.15 * avg_len * (1.0 + 0.05 * np.log1p(round_num))

    for u, v in tour_edges:
        usage = float(U[u, v])
        # Linear scaling: penalty grows with each previous use of this edge.
        # The 0.5 floor ensures even first-use edges (usage=0 after accumulation? 
        # No, usage is count of times used in previous rounds. 
        # In the first call, usage is 0 for all edges if they haven't been used yet?
        # The evaluator adds current tour edges to edge_n_used BEFORE calling the function?
        # "add the current tour's edges to edge_n_used, call your function..."
        # So usage is at least 1 for the current tour edges in the first round.
        # If usage=1, penalty = base * (0.5 + 0.8*1) = 1.3 * base.
        # If usage=20, penalty = base * (0.5 + 0.8*20) = 16.5 * base.
        ep = base_coeff * (0.5 + 0.8 * usage)
        penalty[u, v] += ep
        penalty[v, u] += ep

    # Global penalty for frequently-used edges NOT in the current tour.
    # This discourages the solver from re-forming tours that share many
    # high-usage edges with past local optima.
    # Scale linearly with usage as well.
    global_coeff = 0.02 * avg_len

    # Mask: True where NOT a current-tour edge (and not diagonal)
    mask = np.ones((n, n), dtype=bool)
    for u, v in tour_edges:
        mask[u, v] = False
        mask[v, u] = False
    np.fill_diagonal(mask, False)

    # Apply global penalty only where mask is True and usage > 0
    # Use raw usage count U
    global_pen = np.where(mask & (U > 0),
                          global_coeff * U, 0.0)
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
