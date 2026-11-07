"""Guided local search penalty rule for the TSP – geometrically-aware repulsion."""
import numpy as np
import time

_START_TIME = time.time()
_TIME_LIMIT = 320.0  # Safety margin before 330s kill


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Guided local search penalty rule for TSP.
    
    Two-tier penalty:
    1. Current tour edges get a very large penalty (near-hard-constraint)
       to force 2-opt to explore a diverse neighborhood.
    2. Non-tour edges get a soft geometrically-aware penalty based on
       usage count and rank among incident edges.
    
    The tour-edge penalty decays over rounds; early rounds break more edges.
    """
    # Time guard
    elapsed = time.time() - _START_TIME
    if elapsed > _TIME_LIMIT:
        return edge_distance.copy()

    n = edge_distance.shape[0]
    tour = local_opt_tour

    if n < 2 or len(tour) < 2:
        return edge_distance.copy()

    D = np.asarray(edge_distance, dtype=np.float64)
    U = np.asarray(edge_n_used, dtype=np.float64)

    # --- Compute current tour length and identify tour edges ---
    tour_len = 0.0
    tour_edge_mask = np.zeros((n, n), dtype=bool)
    for i in range(len(tour)):
        u = tour[i]
        v = tour[(i + 1) % len(tour)]
        tour_len += D[u, v]
        tour_edge_mask[u, v] = True
        tour_edge_mask[v, u] = True

    # --- Infer round number ---
    max_used = int(U.max()) if U.size > 0 else 0
    round_num = max_used + 1  # 1-based

    # --- Tour edge penalty: large, decaying over rounds ---
    # Early rounds: very aggressive breaking. Late rounds: moderate.
    # Use exponential decay: strong initially, settles to a baseline.
    tour_penalty_scale = 1e6 * (D.max() / max(n, 1)) * (1.0 + 0.5 * np.exp(-0.2 * round_num))
    # The 0.5*exp term adds a mild early boost; base is 1e6*avg_dist which is huge
    
    # --- Compute geometric rank factor for non-tour edges ---
    # For each node, rank its incident edges by distance.
    # rank_factor[i,j] = 1/(1 + rank) so shortest incident edge gets ~1, longest ~1/n
    rank_factor = np.zeros((n, n), dtype=np.float64)
    
    for i in range(n):
        row = D[i, :].copy()
        row[i] = np.inf
        order = np.argsort(row)
        ranks = np.empty(n, dtype=np.float64)
        ranks[order] = np.arange(n, dtype=np.float64)
        ranks[i] = 0.0
        rf = 1.0 / (1.0 + ranks)
        rank_factor[i, :] = rf
        rank_factor[:, i] = rf
    
    # Symmetrize
    rank_factor = (rank_factor + rank_factor.T) * 0.5
    np.fill_diagonal(rank_factor, 0.0)

    # --- Soft usage for non-tour edges ---
    soft_U = U / (1.0 + U)
    np.fill_diagonal(soft_U, 0.0)

    # --- Non-tour edge penalty: gentle, usage-weighted, rank-weighted ---
    # Scale relative to average edge length
    avg_edge = tour_len / max(len(tour), 1)
    # Decay: early rounds use slightly more perturbation on non-tour edges too
    k_non_tour = 0.3 + 0.2 * np.exp(-0.15 * round_num)
    non_tour_penalty = k_non_tour * avg_edge * soft_U * rank_factor

    # --- Build result ---
    result = D.copy()
    
    # Apply tour edge penalty (large)
    result[tour_edge_mask] += tour_penalty_scale
    
    # Apply non-tour edge penalty (small)
    result += non_tour_penalty

    # Ensure symmetry
    result = (result + result.T) * 0.5
    # Ensure finite
    result = np.nan_to_num(result, nan=0.0,
                           posinf=np.finfo(np.float64).max / 10,
                           neginf=0.0)
    # Zero diagonal
    np.fill_diagonal(result, 0.0)

    return result
