import numpy as np
import time

def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    start_time = time.time()
    deadline = start_time + 280.0  # 300s limit, 20s safety margin
    
    n = edge_distance.shape[0]
    if n == 0:
        return edge_distance.copy()

    # Initialize with true distances
    d_out = edge_distance.copy()
    
    # Find max distance to scale penalties
    max_d = np.max(d_out)
    if max_d == 0:
        return d_out

    # 1. Build tour mask
    tour_mask = np.zeros((n, n), dtype=float)
    try:
        for i in range(n):
            u = local_opt_tour[i]
            v = local_opt_tour[(i + 1) % n]
            if 0 <= u < n and 0 <= v < n:
                tour_mask[u, v] = 1.0
                tour_mask[v, u] = 1.0
    except Exception:
        return edge_distance.copy()

    # 2. Cumulative penalty
    # Increased alpha to break local optima more effectively
    cum_sat = edge_n_used / (1.0 + edge_n_used)
    alpha_cum = 0.8
    penalty = alpha_cum * max_d * cum_sat
    
    # 3. Current tour penalty
    alpha_curr = 0.8
    penalty += alpha_curr * max_d * tour_mask
    
    # Add penalty to distances
    d_out += penalty
    
    # Safety: ensure finite
    if not np.all(np.isfinite(d_out)):
        return edge_distance.copy()
        
    return d_out
