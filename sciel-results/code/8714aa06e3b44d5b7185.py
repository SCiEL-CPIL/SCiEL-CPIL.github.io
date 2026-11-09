"""Guided local search penalty for TSP; Hybrid additive global history repulsion and multiplicative current tour escape."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Applies a hybrid penalty strategy:
    1. Additive penalty to all edges based on historical usage (edge_n_used) to prevent revisiting bad edges.
    2. Multiplicative penalty to edges in the current local_opt_tour to force 2-opt to break them.
    
    This differs from the previous strategy by:
    - Penalizing ALL edges globally (not just current tour) to leverage full history.
    - Using additive penalties for history to avoid distorting the relative geometry of the entire matrix too much.
    - Using multiplicative penalties for current tour edges to ensure strong local escape.
    
    Args:
    edge_distance: np.ndarray, true distance matrix.
    local_opt_tour: list or np.ndarray, the current tour as a sequence of node indices.
    edge_n_used: np.ndarray, symmetric matrix of usage counts.
    
    Returns:
    np.ndarray: Modified distance matrix for the next 2-opt phase.
    """
    n = len(local_opt_tour)
    if n < 2:
        return edge_distance.copy()
    
    # Copy the true distances
    new_dist = edge_distance.copy()
    
    # 1. Global Additive Penalty based on history
    # We want to discourage edges that have been used frequently in the past.
    # The penalty should be proportional to the count, but capped to avoid extreme values.
    # Using an additive penalty preserves the relative ordering of edges that are not penalized,
    # which helps 2-opt maintain good structure in unpenalized regions.
    
    # Determine a scale factor for the additive penalty.
    # The typical edge length in a unit square TSP is approx 1/sqrt(n) for nearest neighbor?
    # Or just use the mean distance.
    mean_dist = np.mean(edge_distance)
    
    # Max usage count to cap the effect
    max_count = np.max(edge_n_used) if edge_n_used.size > 0 else 0
    if max_count == 0:
        max_count = 1
    
    # Scale the additive penalty so that an edge used 'max_count' times gets a penalty
    # comparable to a significant fraction of the mean distance (e.g., 50%).
    # This prevents the penalty from dominating the geometry completely.
    penalty_scale = 0.5 * mean_dist / max_count
    
    # Apply additive penalty to all edges where count > 0
    # We do this vectorized for speed
    mask = edge_n_used > 0
    if np.any(mask):
        # Penalty = count * scale
        # We add this to the distance
        new_dist[mask] += edge_n_used[mask] * penalty_scale
        # Ensure symmetry (edge_n_used is symmetric, so this is already handled if we apply to full matrix)
        # But just in case, let's enforce symmetry on the result
        # new_dist = (new_dist + new_dist.T) / 2.0 -- No, this might average out asymmetric errors if any.
        # The input is symmetric, so new_dist should remain symmetric if we apply the same mask.
        
    # 2. Local Multiplicative Penalty on Current Tour Edges
    # We want to force the 2-opt to break the current tour.
    # We apply a multiplicative factor to edges currently in the tour.
    # The factor should be stronger for edges that are also historically "bad" (high count).
    
    for i in range(n):
        u = local_opt_tour[i]
        v = local_opt_tour[(i + 1) % n]
        
        count = edge_n_used[u, v]
        
        # Base penalty for being in the current tour
        # Even if count is 0, we want to encourage changing this edge to escape the local optimum.
        # If count > 0, we want an even stronger penalty.
        
        # Factor: 1.2 for new edges, increasing to 2.0 for heavily used edges.
        # Let's use a smooth function.
        # factor = 1.0 + 0.1 * min(count, 10)
        
        # If the edge has never been used (count=0), we still penalize it slightly to encourage exploration.
        # If it has been used, penalize more.
        
        capped_count = min(count, 10.0)
        factor = 1.2 + 0.08 * capped_count
        
        # Apply multiplicative penalty
        # Note: new_dist[u,v] already has the additive penalty.
        # Multiplying it will amplify the additive part too.
        # This is desirable: historically bad edges in the current tour get hit hardest.
        
        new_dist[u, v] *= factor
        new_dist[v, u] *= factor

    # Final sanity check: ensure finite values and symmetry
    # The operations are safe, but let's verify symmetry just in case of floating point drift
    # new_dist = (new_dist + new_dist.T) / 2.0
    
    return new_dist
# EVOLVE-BLOCK-END
