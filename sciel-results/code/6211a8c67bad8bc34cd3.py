"""Rewritten program for guided local search penalty rule (TSP)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Guided local search penalty for TSP.
    
    Combines:
    1. A mild global soft penalty proportional to edge usage, which gently
       discourages frequently-used edges across the whole matrix.
    2. A stronger targeted penalty on edges in the current local-optimal tour,
       scaled by their usage count, to specifically break the current solution.
    
    The penalty is multiplicative and bounded to keep the modified distance
    matrix well-conditioned for 2-opt.
    """
    # Deadline guard: track elapsed time to avoid timeout issues.
    t0 = time.time()
    
    n = edge_distance.shape[0]
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
    
    # Global soft penalty: apply a small penalty to all edges based on usage.
    # This helps the search escape by making previously-used edges less attractive.
    # Use a mild multiplicative factor so the landscape is gently reshaped.
    global_penalty = 1.0 + 0.02 * (edge_n_used / max_used)
    penalized *= global_penalty
    
    # Targeted penalty on current tour edges.
    # These edges are the ones we most want to break so 2-opt can find new structure.
    for i in range(m):
        u = tour[i]
        v = tour[(i + 1) % m]
        used = edge_n_used[u, v]
        # Penalty: base multiplier plus scaling with usage.
        # Higher usage -> higher penalty, encouraging the search to avoid stale edges.
        tour_penalty = 1.25 + 0.35 * (used / max_used) + 0.015 * used
        # Cap to avoid extreme values that could distort 2-opt
        tour_penalty = min(tour_penalty, 4.0)
        penalized[u, v] *= tour_penalty
        penalized[v, u] *= tour_penalty
    
    # Safety: ensure all values are finite
    if not np.all(np.isfinite(penalized)):
        penalized = np.nan_to_num(penalized, nan=0.0, posinf=1e10, neginf=0.0)
    
    # Time guard: if we're taking too long (shouldn't happen for n=100), return anyway
    elapsed = time.time() - t0
    # This function should take microseconds to milliseconds
    
    return penalized
# EVOLVE-BLOCK-END
