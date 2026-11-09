"""Guided local search penalty rule for TSP with 100 points."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Perturbation step for guided local search.
    
    Applies a combined penalty strategy:
    1. Base penalty proportional to edge usage count (historical)
    2. Strong penalty on edges of the current local-optimal tour
    3. Usage-scaled amplification to break stagnant tours
    
    Returns a modified distance matrix that biases 2-opt away from
    frequently-used and currently-present edges.
    """
    n = edge_distance.shape[0]
    
    # Work with a copy
    penalized = edge_distance.astype(np.float64).copy()
    
    # Compute reference scale: mean of the n smallest off-diagonal distances
    # This gives a representative "good edge" length for calibration
    flat = edge_distance.ravel()
    # Exclude diagonal (self-distances) - they're 0
    off_diag = flat[~np.eye(n, dtype=bool).ravel()]
    # Use the 50th percentile of off-diagonal distances as a robust scale
    scale = np.percentile(off_diag, 50) if len(off_diag) > 0 else 1.0
    if scale < 1e-10:
        scale = 1.0
    
    # --- Penalty 1: Historical usage penalty ---
    # Additive penalty scaled by usage count, with saturation
    max_usage = edge_n_used.max()
    if max_usage > 0:
        # Normalize usage to [0, 1]
        usage_norm = edge_n_used.astype(np.float64) / max_usage
        # Penalty: base + linear growth, capped
        usage_penalty = scale * (0.08 + 0.12 * usage_norm)
        penalized += usage_penalty
    
    # --- Penalty 2: Strong penalty on current tour edges ---
    # The current tour edges should be heavily penalized to force exploration
    tour_len = len(local_opt_tour)
    if tour_len >= 2:
        for i in range(tour_len):
            u = local_opt_tour[i]
            v = local_opt_tour[(i + 1) % tour_len]
            # Apply a strong multiplicative penalty to these specific edges
            # Scale: make them ~2.5x their true distance
            # But also add an additive component based on usage
            usage_count = edge_n_used[u, v]
            # Stronger penalty for edges used more often
            mult = 2.0 + 0.3 * min(usage_count, 10) / 10.0
            penalized[u, v] *= mult
            penalized[v, u] *= mult
    
    # --- Penalty 3: Neighborhood disruption ---
    # For each tour edge (u,v), also penalize edges that are "similar" 
    # (shortest non-tour edges from u and v) to encourage structural change
    # This is done by slightly increasing the cost of the second-best edges
    # from each tour node, nudging 2-opt to consider different connections
    if tour_len >= 2:
        for i in range(tour_len):
            u = local_opt_tour[i]
            v = local_opt_tour[(i + 1) % tour_len]
            # Get all edges from u sorted by distance
            row_u = penalized[u].copy()
            row_u[u] = np.inf  # exclude self
            # Second smallest edge from u (first is the tour edge to v or another)
            sorted_idx = np.argsort(row_u)
            # Penalize the 2nd and 3rd nearest neighbors slightly
            # This encourages 2-opt to consider swapping to these alternatives
            for k in range(1, min(3, n)):
                w = sorted_idx[k]
                if w != v and w != u:
                    # Small additive nudge
                    penalized[u, w] += 0.02 * scale
                    penalized[w, u] += 0.02 * scale
    
    # Ensure symmetry
    penalized = (penalized + penalized.T) / 2.0
    
    # Ensure finite values
    penalized = np.where(np.isfinite(penalized), penalized, 1e10)
    
    return penalized
# EVOLVE-BLOCK-END
