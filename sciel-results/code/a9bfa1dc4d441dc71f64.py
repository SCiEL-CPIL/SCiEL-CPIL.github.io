"""Guided local search penalty rule for TSP with adaptive kick and edge-specific penalties."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Module-level state for tracking rounds and timing
_round_counter = 0
_start_time = None
_DEADLINE_FRACTION = 0.9  # Use 90% of the 330s budget


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Returns a modified distance matrix for the evaluator's 2-opt.
    
    Strategy:
    1. Track the round number to adapt penalty strength.
    2. Apply a kick to edges in the current local optimum tour.
    3. The kick magnitude increases with usage count to encourage diversification.
    4. Reward under-used edges with a small negative offset.
    5. Use smooth, bounded penalties to avoid distorting the metric too much.
    """
    global _round_counter, _start_time
    
    # Initialize timing if not already done
    if _start_time is None:
        _start_time = time.monotonic()
    
    # Safety check: if we're running out of time, return the original distances
    elapsed = time.monotonic() - _start_time
    if elapsed > 330.0 * _DEADLINE_FRACTION:
        return edge_distance.copy()
    
    # Increment round counter
    _round_counter += 1
    
    n = edge_distance.shape[0]
    
    # Extract edges from the current tour
    tour = local_opt_tour
    tour_edges = set()
    for i in range(n):
        u = tour[i]
        v = tour[(i + 1) % n]
        if u < v:
            tour_edges.add((u, v))
        else:
            tour_edges.add((v, u))
    
    # Create the modified distance matrix
    modified_dist = edge_distance.copy()
    
    # Get statistics for normalization
    max_used = edge_n_used.max()
    if max_used == 0:
        max_used = 1
    
    # Mean edge distance for scale-invariant penalties
    mean_dist = np.mean(edge_distance)
    if mean_dist == 0:
        mean_dist = 1.0
    
    # Adaptive base penalty strength
    # Decay in later rounds to allow convergence
    round_decay = np.exp(-0.05 * _round_counter)
    base_penalty = 0.4 * round_decay
    
    # Penalty for edges in the current tour
    for i in range(n):
        u = tour[i]
        v = tour[(i + 1) % n]
        count = edge_n_used[u, v]
        
        # Smooth penalty: tanh-based to bound the effect
        normalized_usage = count / max_used
        smooth_penalty = np.tanh(2.0 * normalized_usage)
        
        # Scale by mean distance for consistency
        kick = base_penalty * smooth_penalty * mean_dist
        
        # Apply penalty to both directions (symmetric)
        modified_dist[u, v] += kick
        modified_dist[v, u] += kick
    
    # Reward under-used edges NOT in the current tour
    # This encourages exploration of less-traveled edges
    reward_strength = 0.1 * round_decay
    for i in range(n):
        for j in range(i + 1, n):
            if (i, j) not in tour_edges and (j, i) not in tour_edges:
                count = edge_n_used[i, j]
                # Under-used edges get a reward (negative offset)
                if count < 0.3 * max_used:
                    # Scale reward by how under-used the edge is
                    underuse_factor = 1.0 - (count / (0.3 * max_used)) if max_used > 0 else 1.0
                    reward = reward_strength * underuse_factor * mean_dist
                    modified_dist[i, j] -= reward
                    modified_dist[j, i] -= reward
    
    # Ensure the matrix remains finite and symmetric
    modified_dist = np.maximum(modified_dist, 0.01)  # Keep positive to avoid negative distances
    modified_dist = (modified_dist + modified_dist.T) / 2.0
    
    # Verify finiteness
    if not np.all(np.isfinite(modified_dist)):
        return edge_distance.copy()
    
    return modified_dist
# EVOLVE-BLOCK-END
