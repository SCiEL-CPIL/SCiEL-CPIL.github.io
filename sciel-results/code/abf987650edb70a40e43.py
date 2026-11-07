"""Guided local search penalty rule for the TSP."""
# EVOLVE-BLOCK-START
import numpy as np
import time


_START_TIME = time.time()
_TIME_LIMIT = 310.0  # Safety margin before 330s kill


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Guided local search perturbation: hybrid multiplicative/additive penalty.
    
    - Multiplicative penalty on tour edges, scaled by usage (changes relative weights).
    - Light additive penalty on all edges based on usage (global exploration).
    - Adaptive coefficient that grows with round number.
    - Soft-saturating function for smooth, bounded penalties.
    """
    # Time guard
    if time.time() - _START_TIME > _TIME_LIMIT:
        return edge_distance.copy()

    n = edge_distance.shape[0]
    if n < 2:
        return edge_distance.copy()

    D = np.asarray(edge_distance, dtype=np.float64)
    U = np.asarray(edge_n_used, dtype=np.float64)
    tour = local_opt_tour

    # Compute mean distance for scaling
    triu_idx = np.triu_indices(n, k=1)
    mean_dist = np.mean(D[triu_idx])
    if mean_dist < 1e-12:
        mean_dist = 1.0

    # Soft-saturating usage factor
    sat = U / (1.0 + U)

    # Infer round number from max usage
    global_max = float(U.max()) if U.size > 0 else 0.0
    round_num = max(1, int(round(global_max)))

    # Adaptive multiplicative coefficient: grows with round
    # Round 1: ~0.25, Round 10: ~0.45, Round 20: ~0.65
    a = min(0.7, 0.25 + 0.02 * (round_num - 1))

    # Build the modified distance matrix (start with true distances)
    modified = D.copy()

    # Light additive global penalty on all edges based on usage
    global_scale = 0.03 * mean_dist
    modified += global_scale * sat

    # Strong multiplicative penalty on edges in the current tour
    # This changes relative weights, breaking local optima more effectively
    for i in range(n):
        j = tour[i]
        # Multiplicative: multiply by (1 + a * sat)
        # sat is in [0,1), so penalty factor is in [1, 1+a)
        factor = 1.0 + a * sat[i, j]
        modified[i, j] *= factor
        modified[j, i] *= factor

    # Ensure symmetry
    modified = (modified + modified.T) * 0.5

    # Ensure finite
    modified = np.nan_to_num(modified, nan=0.0,
                             posinf=np.finfo(np.float64).max / 10,
                             neginf=0.0)
    # Zero diagonal
    np.fill_diagonal(modified, 0.0)

    return modified
# EVOLVE-BLOCK-END
