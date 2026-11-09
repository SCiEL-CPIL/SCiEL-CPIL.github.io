"""Rewritten program for guided local search penalty rule (TSP)."""
# EVOLVE-BLOCK-START
import numpy as np


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Guided local search penalty for TSP.
    
    Applies targeted penalties to edges in the current local-optimal tour,
    scaled by their usage count. Also applies a mild global penalty to
    discourage overused edges.
    """
    # Make a copy so we don't mutate the input
    penalized = edge_distance.copy()
    
    # Extract edges from the current tour
    tour = np.asarray(local_opt_tour)
    m = len(tour)
    
    if m < 2:
        return penalized
    
    # Get max usage for normalization
    max_used = edge_n_used.max()
    if max_used <= 0:
        max_used = 1.0
    
    # Mild global penalty for overused edges
    global_penalty = 1.0 + 0.03 * (edge_n_used / max_used)
    penalized *= global_penalty
    
    # Targeted penalty on current tour edges
    for i in range(m):
        u = tour[i]
        v = tour[(i + 1) % m]
        used = edge_n_used[u, v]
        # Penalty scales with usage: more used edges get penalized more
        # Base multiplier of 1.3, plus additional scaling
        tour_penalty = 1.3 + 0.4 * (used / max_used) + 0.02 * used
        # Cap to avoid extreme values
        tour_penalty = min(tour_penalty, 5.0)
        penalized[u, v] *= tour_penalty
        penalized[v, u] *= tour_penalty
    
    # Safety: ensure all values are finite
    if not np.all(np.isfinite(penalized)):
        penalized = np.nan_to_num(penalized, nan=0.0, posinf=1e10, neginf=0.0)
    
    return penalized
# EVOLVE-BLOCK-END
