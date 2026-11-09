"""Rewritten program for guided local search penalty rule (TSP)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Guided local search penalty for TSP.
    
    Combines:
    1. Global soft penalty proportional to edge usage (from inspiration 1),
       which gently discourages frequently-used edges across the whole matrix.
    2. Stronger targeted penalty on edges in the current local-optimal tour,
       scaled by their usage count, to specifically break the current solution.
    
    This dual approach ensures the search is pushed away from both the current
    local optimum and any edges that have been over-used historically.
    """
    t0 = time.time()
    
    n = edge_distance.shape[0]
    penalized = edge_distance.copy()
    
    tour = np.asarray(local_opt_tour)
    m = len(tour)
    
    if m < 2:
        return penalized
    
    # --- Part 1: Global soft penalty based on usage ---
    # This gently increases the cost of edges that have been used many times,
    # encouraging the search to explore less-traveled parts of the graph.
    max_used = edge_n_used.max()
    if max_used > 0:
        # Normalized usage in [0, 1], scaled by a small factor
        global_penalty = 1.0 + 0.1 * edge_n_used / (1.0 + max_used)
        penalized *= global_penalty
    
    # --- Part 2: Stronger targeted penalty on current tour edges ---
    # For each edge in the current tour, apply a stronger multiplicative penalty
    # that grows with usage. This directly attacks the current local optimum.
    for i in range(m):
        u = tour[i]
        v = tour[(i + 1) % m]
        used = edge_n_used[u, v]
        
        # Penalty factor: base boost plus usage-scaled component
        # The 0.5 * (used/max_used) term normalizes usage
        # The 0.1 * used term adds an absolute usage penalty
        if max_used > 0:
            penalty_factor = 1.0 + 0.5 * (used / max_used) + 0.1 * used
        else:
            penalty_factor = 1.5  # Minimum penalty when no usage recorded
        
        # Cap to avoid extreme values
        penalty_factor = min(penalty_factor, 5.0)
        
        penalized[u, v] *= penalty_factor
        penalized[v, u] *= penalty_factor
    
    # Safety: ensure all values are finite
    if not np.all(np.isfinite(penalized)):
        penalized = np.nan_to_num(penalized, nan=0.0, posinf=1e10, neginf=0.0)
    
    # Time guard: return immediately if taking too long (shouldn't happen)
    elapsed = time.time() - t0
    if elapsed > 0.5:  # Should be microseconds, this is just a safeguard
        pass
    
    return penalized
# EVOLVE-BLOCK-END
