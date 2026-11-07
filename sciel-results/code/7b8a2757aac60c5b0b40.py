"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_DEADLINE = 325.0  # safety margin before 330s kill


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Guided local search penalty rule for TSP.
    
    Combines:
    1. Historical usage penalty (smooth, saturating)
    2. Current-tour edge penalty to escape local optima
    3. Edge-length-aware scaling: longer edges get stronger relative penalties
       since they are less valuable to preserve
    """
    n = len(local_opt_tour)
    if n <= 2:
        return edge_distance.copy()
    
    # Deadline check
    if time.time() > _DEADLINE:
        return edge_distance.copy()
    
    # Work on a copy
    modified = edge_distance.copy()
    
    # Extract current tour edges
    tour_edges = []
    for i in range(n):
        u = local_opt_tour[i]
        v = local_opt_tour[(i + 1) % n]
        tour_edges.append((u, v))
    
    # Base penalty from edge usage history (smooth saturation)
    max_used = edge_n_used.max()
    if max_used > 0:
        # Normalize usage counts
        norm_used = edge_n_used / (1.0 + max_used)
        # Smooth penalty: use log-sum-exp style soft saturation
        # penalty = alpha * tanh(beta * norm_used) for smooth bounded penalty
        alpha_hist = 0.15
        beta_hist = 1.5
        hist_penalty = alpha_hist * np.tanh(beta_hist * norm_used)
    else:
        hist_penalty = np.zeros_like(edge_n_used)
    
    # Current tour edge penalty: strongly penalize edges in the current local optimum
    # to force escape from the current basin
    alpha_tour = 0.30
    tour_penalty = np.zeros_like(edge_n_used)
    for (u, v) in tour_edges:
        tour_penalty[u, v] = alpha_tour
        tour_penalty[v, u] = alpha_tour
    
    # Edge-length-aware scaling: longer edges get stronger relative penalties
    # This differentiates between short and long edges, applying stronger penalties
    # to longer edges that are less valuable to preserve
    # Compute median edge length as reference
    diag_idx = ~np.eye(n, dtype=bool)
    edge_lengths = edge_distance[diag_idx]
    if len(edge_lengths) > 0:
        median_len = np.median(edge_lengths)
        if median_len > 0:
            # Scale factor: edges longer than median get amplified penalty
            # Use smooth scaling: scale = 1 + gamma * (d / median - 1) for d > median
            gamma = 0.5
            d = edge_distance
            # Only apply to off-diagonal
            d_off = d.copy()
            np.fill_diagonal(d_off, 0.0)
            # Compute relative length
            rel_len = d_off / median_len
            # Amplification factor: >1 for long edges, <1 for short edges (but not below 0.5)
            amp = 1.0 + gamma * np.maximum(rel_len - 1.0, 0.0)
            # Cap amplification
            amp = np.minimum(amp, 2.0)
            # Apply only to off-diagonal
            amp[~diag_idx] = amp[~diag_idx]
            np.fill_diagonal(amp, 1.0)
        else:
            amp = np.ones_like(edge_distance)
    else:
        amp = np.ones_like(edge_distance)
    
    # Combine penalties
    total_penalty = hist_penalty + tour_penalty
    # Apply edge-length-aware scaling
    total_penalty = total_penalty * amp
    
    # Apply penalty multiplicatively to distances
    # Use smooth exponential form: d * (1 + penalty)
    # But cap the total multiplier to avoid extreme distortions
    multiplier = 1.0 + total_penalty
    multiplier = np.minimum(multiplier, 3.0)  # Cap at 3x
    np.fill_diagonal(multiplier, 1.0)  # Keep diagonal unchanged
    
    modified = modified * multiplier
    
    # Ensure finite and symmetric
    modified = np.nan_to_num(modified, nan=0.0, posinf=1e6, neginf=0.0)
    modified = (modified + modified.T) / 2.0
    np.fill_diagonal(modified, 0.0)
    
    return modified
# EVOLVE-BLOCK-END
