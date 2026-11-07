"""Guided local search penalty rule for the TSP – dynamic edge removal."""
import numpy as np
import time

_START_TIME = time.time()
_TIME_LIMIT = 320.0  # Safety margin before 330s kill





def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Guided local search penalty rule for TSP using dynamic edge removal.
    
    Edges with usage counts exceeding a round-dependent threshold are assigned
    a large finite distance, forcing 2-opt to reconstruct the tour from fresh
    edges. A smaller additive penalty is also applied to recently-used edges.
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

    # Infer round number from max usage count
    max_used = int(U.max()) if U.size > 0 else 0
    round_num = max_used + 1  # 1-based, capped at 20

    # --- Dynamic edge removal threshold ---
    # Threshold increases with round: early rounds keep most edges,
    # later rounds force more reconstruction.
    # threshold = 2 * round / 20, so at round 20, edges used > 2 times are removed
    threshold = 2.0 * min(round_num, 20) / 20.0

    # Large finite value for "removed" edges
    large_val = 100.0  # Much larger than typical TSP edge lengths (~0.5)

    # --- Identify edges to remove ---
    # Edges with usage > threshold are "removed" (set to large distance)
    # But we must ensure the graph stays connected, so we only remove
    # edges that are not in the current tour (to avoid breaking it immediately)
    # Actually, we should remove based on usage, but keep the current tour intact
    # for the first few rounds, then gradually remove tour edges too.

    # For robustness, let's remove all edges with U > threshold,
    # but if that disconnects the graph, we'll handle it via the additive penalty.
    # Since 2-opt operates on the tour, removing edges not in the tour is safe.
    # Removing tour edges forces the tour to change.

    # Strategy:
    # 1. Set edges with U > threshold to large_val (except diagonal)
    # 2. Add a small additive penalty to edges with U > 0

    result = D.copy()

    # Create a mask of edges to "remove"
    remove_mask = (U > threshold) & (U > 0)
    np.fill_diagonal(remove_mask, False)

    # Apply removal: set to large_val
    result[remove_mask] = large_val

    # --- Additive penalty for recently used edges ---
    # Small penalty to discourage immediate reuse, even if not removed
    # Scale by usage count with saturation
    soft_U = U / (1.0 + U)
    # Penalty coefficient: starts moderate, decays slightly
    pen_coeff = 0.5 * (1.0 - 0.02 * min(round_num, 20))
    additive_pen = pen_coeff * soft_U
    # Apply only to non-removed edges
    additive_pen[remove_mask] = 0.0
    result += additive_pen

    # Ensure symmetry
    result = (result + result.T) * 0.5

    # Ensure finite
    result = np.nan_to_num(result, nan=0.0,
                           posinf=np.finfo(np.float64).max / 10,
                           neginf=0.0)
    # Zero diagonal
    np.fill_diagonal(result, 0.0)

    return result
