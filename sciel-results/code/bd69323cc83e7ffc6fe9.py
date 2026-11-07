# EVOLVE-BLOCK-START
import numpy as np
import time

# Global start time to enforce wall-clock limit across multiple calls
_GLOBAL_START_TIME = time.time()
_WALL_CLOCK_LIMIT = 320.0  # Safety margin before 330s kill

def _safe_update(edge_distance, local_opt_tour, edge_n_used):
    """
    Guided local search penalty rule for TSP.
    
    Strategy:
    1. Infer round number from edge_n_used max value.
    2. Apply adaptive additive penalty that scales with usage and round.
    3. Stronger penalty on edges in the current local-opt tour.
    4. Weaker global penalty on other frequently used edges.
    5. Ensure the resulting matrix is finite, symmetric, and has zero diagonal.
    """
    # Check time limit
    if time.time() - _GLOBAL_START_TIME > _WALL_CLOCK_LIMIT:
        return edge_distance.copy()
    
    n = edge_distance.shape[0]
    if n < 2 or len(local_opt_tour) < 2:
        return edge_distance.copy()
    
    D = np.asarray(edge_distance, dtype=np.float64)
    U = np.asarray(edge_n_used, dtype=np.float64)
    
    # If no edges have been used yet, return original
    if U.max() == 0:
        return D.copy()
    
    # Infer round: max edge usage indicates how many rounds have passed
    max_used = int(U.max())
    round_num = max_used
    
    # Extract tour edges and compute scale
    tour_edges = []
    tour_lens = []
    tour_mask = np.zeros((n, n), dtype=bool)
    for i in range(len(local_opt_tour)):
        u = local_opt_tour[i]
        v = local_opt_tour[(i + 1) % len(local_opt_tour)]
        tour_edges.append((u, v))
        tour_lens.append(D[u, v])
        tour_mask[u, v] = True
        tour_mask[v, u] = True
    
    avg_len = float(np.mean(tour_lens)) if tour_lens else 1.0
    if avg_len < 1e-9:
        avg_len = 1.0
    
    # Compute mean distance for scaling
    mask_non_diag = ~np.eye(n, dtype=bool)
    mean_dist = np.mean(D[mask_non_diag]) if n > 1 else avg_len
    
    # Soft saturation for usage: u/(1+u) in [0, 1)
    soft_usage = U / (1.0 + U)
    
    # Adaptive coefficients
    # Strong penalty for current tour edges
    alpha_tour = 0.15 * avg_len * (1.0 + 0.1 * np.log1p(round_num + 1))
    # Weaker penalty for other frequently used edges
    alpha_global = 0.02 * avg_len * (1.0 + 0.05 * np.log1p(round_num + 1))
    
    # Build penalty matrix
    penalty = np.zeros((n, n), dtype=np.float64)
    
    # Penalty for edges in the current tour
    for u, v in tour_edges:
        usage = float(U[u, v])
        # Edge-specific penalty with soft saturation
        ep = alpha_tour * (0.5 + 0.8 * (usage / (1.0 + usage)))
        penalty[u, v] += ep
        penalty[v, u] += ep
    
    # Global penalty for frequently-used edges NOT in the current tour
    # Mask: True where NOT a current-tour edge (and not diagonal)
    global_mask = ~tour_mask & mask_non_diag
    global_pen = np.where(global_mask & (soft_usage > 0),
                          alpha_global * soft_usage, 0.0)
    penalty += global_pen
    
    # Combine: additive penalty on top of true distances
    new_distances = D + penalty
    
    # Ensure symmetry
    new_distances = (new_distances + new_distances.T) / 2.0
    
    # Ensure diagonal is 0
    np.fill_diagonal(new_distances, 0.0)
    
    # Ensure finiteness and positivity
    new_distances = np.where(np.isfinite(new_distances), new_distances, D)
    new_distances = np.maximum(new_distances, 1e-10)
    np.fill_diagonal(new_distances, 0.0)
    
    return new_distances

def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    try:
        return _safe_update(edge_distance, local_opt_tour, edge_n_used)
    except Exception:
        return edge_distance.copy()
# EVOLVE-BLOCK-END
