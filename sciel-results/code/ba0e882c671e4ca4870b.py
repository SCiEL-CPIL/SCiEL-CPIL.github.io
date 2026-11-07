import numpy as np
import time

def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    start_time = time.time()
    deadline = start_time + 295.0  # 5s safety margin before 300s limit
    
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
    
    # Geometry-aware inverse distance weighting
    # Use inverse distance (1/d) for repulsive force on short edges
    # This creates pressure to escape local optima dominated by short edges
    safe_dist = np.where(edge_distance > 1e-9, edge_distance, 1.0)
    inv_d = 1.0 / safe_dist
    max_inv_d = np.max(inv_d)
    if max_inv_d > 0:
        inv_d_norm = inv_d / max_inv_d
    else:
        inv_d_norm = np.ones_like(edge_distance)
    
    # Zero out diagonal for penalty calculation safety
    np.fill_diagonal(inv_d_norm, 0.0)
    
    # 2. Cumulative penalty with smooth saturation
    # Use x / (1 + x) which grows slowly and saturates at 1
    # This avoids over-penalizing edges that have been used many times
    cum_sat = edge_n_used / (1.0 + edge_n_used)
    
    # Apply inverse distance weighting to the cumulative penalty
    # Higher alpha for stronger pressure on frequently-used short edges
    alpha_cum = 0.6
    penalty = alpha_cum * max_d * cum_sat * inv_d_norm
    
    # 3. Current tour penalty
    # Moderate penalty for edges currently in the local optimum to encourage 2-opt to explore
    alpha_curr = 0.2
    penalty += alpha_curr * max_d * tour_mask * inv_d_norm
    
    # Add penalty to distances
    d_out += penalty
    
    # Ensure symmetry (TSP is symmetric)
    d_out = (d_out + d_out.T) / 2.0
    
    # Safety: ensure finite
    if not np.all(np.isfinite(d_out)):
        return edge_distance.copy()
        
    return d_out
