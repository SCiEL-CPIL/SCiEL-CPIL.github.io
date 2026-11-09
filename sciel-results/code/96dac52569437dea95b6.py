"""Guided local search penalty rule for TSP with 100 points."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Perturbation step for guided local search.
    
    Strategy:
    1. Additive usage-based penalty: edges used more often get larger penalties
    2. Strong penalty on current tour edges to force 2-opt to break them
    3. Smooth soft-threshold penalty on near-tour edges to disrupt local structure
    4. All penalties are additive to preserve 2-opt's distance-metric assumptions
    
    Returns a modified distance matrix that biases 2-opt away from
    frequently-used and currently-present edges.
    """
    t0 = time.time()
    DEADLINE = 300.0  # seconds
    SAFETY_MARGIN = 5.0  # seconds before deadline
    
    n = edge_distance.shape[0]
    tour_len = len(local_opt_tour)
    
    # Work with a copy
    penalized = edge_distance.astype(np.float64).copy()
    
    # Compute reference scale: median of off-diagonal distances
    off_diag_mask = ~np.eye(n, dtype=bool)
    off_diag = edge_distance[off_diag_mask]
    scale = np.median(off_diag) if len(off_diag) > 0 else 1.0
    if scale < 1e-10:
        scale = 1.0
    
    # --- Penalty 1: Historical usage penalty (additive, smooth) ---
    max_usage = edge_n_used.max()
    if max_usage > 0:
        # Normalize usage to [0, 1]
        usage_norm = edge_n_used.astype(np.float64) / max_usage
        # Smooth penalty: base + growth, using tanh for bounded smoothness
        # This gives ~0.1*scale for unused edges, up to ~0.25*scale for max-used
        usage_penalty = scale * (0.1 + 0.15 * np.tanh(usage_norm))
        penalized += usage_penalty
    
    # --- Penalty 2: Strong penalty on current tour edges ---
    # These are the edges we want 2-opt to break
    if tour_len >= 2:
        for i in range(tour_len):
            u = local_opt_tour[i]
            v = local_opt_tour[(i + 1) % tour_len]
            # Additive penalty: make tour edges ~1.5x more expensive
            # Scale with usage: more-used edges get stronger penalty
            usage_count = edge_n_used[u, v]
            # Smooth: 1.5 + 0.5*tanh(usage/5) gives 1.5 to 2.0 multiplier
            penalty = scale * (1.5 + 0.5 * np.tanh(usage_count / 5.0))
            penalized[u, v] += penalty
            penalized[v, u] += penalty
    
    # --- Penalty 3: Disrupt near-tour edges using smooth surrogate ---
    # For each tour node, penalize the 2nd and 3rd nearest neighbors
    # This forces 2-opt to consider more distant alternatives
    if tour_len >= 2:
        # Temperature for soft threshold (smooth transition)
        temp = 0.1 * scale
        
        for i in range(tour_len):
            u = local_opt_tour[i]
            v = local_opt_tour[(i + 1) % tour_len]
            
            # Get distances from u to all other nodes (excluding self)
            row_u = penalized[u].copy()
            row_u[u] = np.inf  # exclude self
            
            # Find the 2nd and 3rd nearest neighbors (excluding tour partner v)
            # We'll penalize edges to these nodes to disrupt local structure
            
            # Create a mask: exclude self and tour partner
            exclude_mask = np.zeros(n, dtype=bool)
            exclude_mask[u] = True
            exclude_mask[v] = True
            
            # Get finite distances
            finite_mask = np.isfinite(row_u) & ~exclude_mask
            if np.sum(finite_mask) < 3:
                continue
            
            # Get the distances of candidates
            candidate_dists = row_u[finite_mask]
            
            # Find 2nd and 3rd smallest (indices in the candidate set)
            sorted_idx = np.argsort(candidate_dists)
            if len(sorted_idx) >= 3:
                # 2nd and 3rd nearest (0-indexed: 1 and 2)
                second_nearest_idx = sorted_idx[1]
                third_nearest_idx = sorted_idx[2]
                
                # Get actual node indices
                candidate_nodes = np.where(finite_mask)[0]
                second_node = candidate_nodes[second_nearest_idx]
                third_node = candidate_nodes[third_nearest_idx]
                
                # Apply smooth penalty: edges close to the minimum get less penalty
                # Edges farther from minimum get more penalty (up to a cap)
                min_dist = candidate_dists[sorted_idx[0]]
                
                for node, weight in [(second_node, 0.08), (third_node, 0.05)]:
                    # Smooth penalty based on relative distance
                    rel_dist = (row_u[node] - min_dist) / (scale + 1e-10)
                    # Tanh gives smooth bounded penalty: 0 to weight*scale
                    penalty = weight * scale * np.tanh(rel_dist / temp)
                    penalized[u, node] += penalty
                    penalized[node, u] += penalty
    
    # Ensure symmetry
    penalized = (penalized + penalized.T) / 2.0
    
    # Ensure finite values
    penalized = np.where(np.isfinite(penalized), penalized, 1e10)
    
    # Final polish: clip to prevent numerical issues
    penalized = np.clip(penalized, 0.0, 1e6 * scale)
    
    # Deadline guard: if we're close to the limit, return immediately
    if time.time() - t0 > DEADLINE - SAFETY_MARGIN:
        return penalized
    
    return penalized
# EVOLVE-BLOCK-END
