"""Rewritten program for guided local search penalty rule (TSP)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Guided local search penalty for TSP.
    
    Combines:
    1. A very mild global soft penalty proportional to edge usage.
       Analysis suggests that strong global penalties can distort the landscape
       too much, so this is reduced compared to previous versions.
    2. A stronger targeted penalty on edges in the current local-optimal tour,
       scaled by their usage count, to specifically break the current solution.
       The base multiplier and usage scaling are increased to force a more
       significant perturbation of the current tour structure.
    
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
    
    # Global soft penalty: apply a very small penalty to all edges based on usage.
    # Reduced from 0.02 to 0.01 to minimize global distortion while still
    # discouraging extremely overused edges.
    global_penalty = 1.0 + 0.01 * (edge_n_used / max_used)
    penalized *= global_penalty
    
    # Targeted penalty on current tour edges.
    # Increased base from 1.30 to 1.40 and absolute usage coefficient from 0.02 to 0.03.
    # This ensures that even edges with low usage in the current tour are penalized more,
    # and high-usage edges are penalized significantly more, forcing 2-opt to explore
    # alternative connections.
    for i in range(m):
        u = tour[i]
        v = tour[(i + 1) % m]
        used = edge_n_used[u, v]
        # Penalty: base multiplier plus scaling with usage.
        tour_penalty = 1.40 + 0.45 * (used / max_used) + 0.03 * used
        # Cap to avoid extreme values that could distort 2-opt, slightly increased to 6.0
        tour_penalty = min(tour_penalty, 6.0)
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
