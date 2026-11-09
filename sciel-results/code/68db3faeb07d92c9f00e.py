import numpy as np
import time

def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Guided local search penalty for TSP using rank-based convex penalties.
    
    This strategy uses a non-linear (convex) function of the normalized rank
    of edge usage counts. This creates a steep barrier against the most
    frequently used edges (high rank) while leaving low-usage edges nearly
    unperturbed. A stronger targeted penalty is applied to the edges of the
    current local-optimal tour to force the search to escape the current
    local minimum.
    """
    t0 = time.time()
    
    n = edge_distance.shape[0]
    penalized = edge_distance.copy()
    
    tour = np.asarray(local_opt_tour)
    m = len(tour)
    
    if m < 2:
        return penalized
    
    # 1. Compute normalized ranks of edge usage counts
    # We only care about unique edges (i, j) with i < j
    i_idx, j_idx = np.triu_indices(n, k=1)
    usage_vals = edge_n_used[i_idx, j_idx]
    
    if len(usage_vals) == 0:
        return penalized
        
    # Sort to find ranks
    # searchsorted with side='right' gives the count of elements <= x
    # This effectively maps 0 to 0, max to 1 (approx)
    sorted_usage = np.sort(usage_vals)
    # Handle case where all usage is 0
    max_usage = usage_vals.max()
    if max_usage == 0:
        # If no edges have been used, ranks are undefined/0.
        # Just return original or mild penalty?
        # If max_usage is 0, it's the first round.
        # We should still penalize the current tour to start diversification.
        # But rank is 0 for all.
        # Let's just apply a flat penalty to tour edges.
        for i in range(m):
            u = tour[i]
            v = tour[(i + 1) % m]
            penalized[u, v] *= 1.2
            penalized[v, u] *= 1.2
        return penalized

    # Calculate normalized rank in [0, 1]
    # Using searchsorted is O(N log N)
    ranks = np.searchsorted(sorted_usage, usage_vals, side='right') / len(usage_vals)
    
    # Reshape ranks back to a matrix (upper triangle only initially)
    rank_matrix = np.zeros((n, n), dtype=np.float64)
    rank_matrix[i_idx, j_idx] = ranks
    rank_matrix[j_idx, i_idx] = ranks
    
    # 2. Global Soft Penalty
    # Convex function of rank: rank^2
    # Low rank (0) -> penalty 1.0
    # High rank (1) -> penalty 1.0 + alpha
    alpha = 0.5
    global_penalty = 1.0 + alpha * (rank_matrix ** 2)
    penalized *= global_penalty
    
    # 3. Targeted Penalty for Current Tour Edges
    # These edges are the ones defining the current local optimum.
    # We apply a stronger penalty, also convex in rank, plus a base offset.
    beta = 1.5
    base_tour_penalty = 0.4
    
    for i in range(m):
        u = tour[i]
        v = tour[(i + 1) % m]
        
        # Get the rank of this specific edge
        r = rank_matrix[u, v]
        
        # Penalty: Base + Beta * Rank^2
        # This ensures even "new" edges in the tour are penalized (base),
        # but "stale" edges (high rank) get a much larger penalty.
        tour_penalty = 1.0 + base_tour_penalty + beta * (r ** 2)
        
        # Cap to prevent extreme distortions
        tour_penalty = min(tour_penalty, 5.0)
        
        penalized[u, v] *= tour_penalty
        penalized[v, u] *= tour_penalty
        
    # Safety: ensure all values are finite
    if not np.all(np.isfinite(penalized)):
        penalized = np.nan_to_num(penalized, nan=0.0, posinf=1e10, neginf=0.0)
        
    # Time guard
    elapsed = time.time() - t0
    # For N=100, this should be very fast (< 1ms)
    
    return penalized
