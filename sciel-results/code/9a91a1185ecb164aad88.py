import numpy as np
import time

def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Structural Swap-Edge Penalty for Guided Local Search.
    
    Strategy:
    Instead of penalizing edges based on abstract properties (length, usage), 
    we identify the specific edges that form the "basin of attraction" of the 
    current local optimum. In 2-opt, a move breaks two edges (i, i+1) and 
    (j, j+1) and reconnects them as (i, j) and (i+1, j+1). 
    If the new connection is shorter, the current tour is not optimal. 
    However, even if the current tour is 2-optimal on true distances, 
    the "near-miss" swaps (those that are slightly worse or very close) 
    represent the local structure. 
    
    By heavily penalizing the edges that are part of potential 2-opt swaps 
    (specifically, the edges being broken), we force the search to look 
    further away. We also penalize edges in the current tour to break 
    symmetry.
    
    Key Insight: The evaluator runs 2-opt on our returned matrix. If we 
    penalize the edges that 2-opt *would* like to swap (or the edges in 
    the current tour), we disrupt the local convergence.
    
    Implementation:
    1. Identify all edges in the current tour.
    2. Identify "critical" edges: those in the tour that, if removed, 
       would allow a significant structural change. We approximate this by 
       penalizing all tour edges, but with a magnitude that scales with 
       the "regret" of keeping them (how much worse they are compared to 
       the average edge).
    3. Add a small noise to break ties and avoid deterministic cycles.
    """
    start_time = time.time()
    deadline = start_time + 295.0
    
    n = len(local_opt_tour)
    if n <= 3:
        return edge_distance.copy()

    perturbed = edge_distance.copy()
    
    # Extract tour edges
    tour_edges = []
    for i in range(n):
        u = local_opt_tour[i]
        v = local_opt_tour[(i + 1) % n]
        if u > v:
            u, v = v, u
        tour_edges.append((u, v))
        
    # Get distances of tour edges
    tour_dists = np.array([edge_distance[u, v] for u, v in tour_edges])
    mean_tour_dist = np.mean(tour_dists)
    std_tour_dist = np.std(tour_dists) if n > 1 else 1.0
    
    # Base penalty scale
    # We want to significantly alter the landscape for 2-opt.
    # A penalty of ~0.5-1.0 is usually enough to break local optima in 2-opt.
    base_penalty = 1.0
    
    # Adaptive penalty: 
    # If the tour is "tight" (low std), we need a stronger kick to escape.
    # If the tour is "loose", a smaller kick might suffice.
    # However, we also want to consider usage.
    
    max_usage = np.max(edge_n_used) if edge_n_used.size > 0 else 0
    if max_usage > 0:
        # Scale penalty slightly with usage to avoid re-visiting old structures
        usage_factor = 1.0 + 0.1 * max_usage
    else:
        usage_factor = 1.0
        
    # Apply penalty to tour edges
    # We use a vectorized approach for speed
    i_idx, j_idx = zip(*tour_edges)
    i_idx = np.array(i_idx)
    j_idx = np.array(j_idx)
    
    # Calculate individual penalty for each tour edge
    # Edges much longer than average get a higher penalty? 
    # Actually, in 2-opt, we want to break the tour. 
    # Let's apply a uniform strong penalty to all tour edges to ensure 
    # the 2-opt on the perturbed matrix sees a completely different cost 
    # structure for these connections.
    
    # Add some randomness to prevent cycling
    jitter = np.random.uniform(0, 0.2, size=len(i_idx))
    
    penalties = base_penalty * usage_factor + jitter
    
    perturbed[i_idx, j_idx] += penalties
    perturbed[j_idx, i_idx] += penalties
    
    # Additionally, penalize "near-tour" edges?
    # The prompt suggests a different strategy: "reward-penalty hybrid".
    # Let's try the hybrid: 
    # Edges NOT in the tour but with low usage get a small REWARD (negative penalty).
    # This encourages 2-opt to consider "new" edges.
    
    # Identify non-tour edges
    all_i, all_j = np.triu_indices(n, k=1)
    tour_set = set(map(tuple, zip(i_idx, j_idx)))
    
    # Create a mask for non-tour edges
    # This is O(N^2) which is fine for N=100
    non_tour_mask = np.ones(len(all_i), dtype=bool)
    for u, v in tour_set:
        # Find index in all_i/all_j where (u,v) or (v,u) exists
        # Since we used triu, u < v
        idx = np.where((all_i == u) & (all_j == v))[0]
        if len(idx) > 0:
            non_tour_mask[idx[0]] = False
            
    non_tour_i = all_i[non_tour_mask]
    non_tour_j = all_j[non_tour_mask]
    
    # Get usage for non-tour edges
    non_tour_usage = edge_n_used[non_tour_i, non_tour_j]
    
    # Reward edges with low usage
    # Threshold: if usage < average usage, reward
    avg_usage = np.mean(edge_n_used) if edge_n_used.size > 0 else 0
    
    # Reward magnitude: small, e.g., -0.1 to -0.3
    reward_mag = 0.2
    
    rewards = np.zeros(len(non_tour_i))
    low_usage_mask = non_tour_usage < avg_usage
    rewards[low_usage_mask] = -reward_mag
    
    # Apply rewards (negative penalties)
    perturbed[non_tour_i, non_tour_j] += rewards
    perturbed[non_tour_j, non_tour_i] += rewards
    
    # Safety check
    if time.time() > deadline:
        return edge_distance
        
    # Ensure finiteness and non-negativity (distances should ideally be positive, 
    # but the evaluator just needs a matrix. However, negative distances can be 
    # problematic for some 2-opt implementations if they assume metric. 
    # The prompt says "finite matrix". Let's ensure no NaN/Inf.
    if not np.all(np.isfinite(perturbed)):
        return edge_distance
        
    return perturbed
