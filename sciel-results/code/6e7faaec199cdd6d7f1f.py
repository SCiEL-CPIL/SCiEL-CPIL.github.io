"""Guided local search penalty rule for the TSP."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Guided local search perturbation: add targeted penalties to edges in the
    current local optimal tour, scaled by a smooth function of their usage count.
    
    Design:
    - Only penalize edges that are in the current tour (targeted, not global)
    - Penalty magnitude grows with usage count using a soft saturating function
    - Scale penalty relative to the mean edge distance for robustness
    - Use a time guard to ensure we return before the 330s deadline
    """
    t0 = time.time()
    DEADLINE = 325.0  # 5s safety margin before 330s kill
    
    n = edge_distance.shape[0]
    perturbed = edge_distance.copy()
    
    # Get the edges in the current tour
    tour = np.asarray(local_opt_tour, dtype=int)
    tour_edges = []
    for i in range(len(tour) - 1):
        a, b = tour[i], tour[i + 1]
        if a != b:
            tour_edges.append((a, b))
    # Close the tour
    if len(tour) > 1:
        a, b = tour[-1], tour[0]
        if a != b:
            tour_edges.append((a, b))
    
    if not tour_edges:
        return perturbed
    
    # Compute mean edge distance (upper triangle, exclude diagonal)
    iu = np.triu_indices(n, k=1)
    mean_dist = edge_distance[iu].mean()
    if mean_dist <= 0:
        return perturbed
    
    # Smooth penalty function: use a soft saturating function of usage count
    # penalty_i = alpha * mean_dist * (1 - exp(-beta * n_used_i))
    # This gives: 0 penalty when n_used=0, approaches alpha*mean_dist as n_used->inf
    # beta controls how quickly the penalty saturates
    
    alpha = 0.8   # Maximum penalty fraction of mean edge distance
    beta = 0.5    # Saturation rate
    
    for (a, b) in tour_edges:
        if a == b:
            continue
        n_used = edge_n_used[a, b]
        # Smooth penalty: saturates at alpha * mean_dist
        penalty = alpha * mean_dist * (1.0 - np.exp(-beta * n_used))
        perturbed[a, b] += penalty
        perturbed[b, a] += penalty
    
    # Time guard: if we're close to deadline, return immediately
    if time.time() - t0 > (DEADLINE - time.time() if False else 2.0):
        return perturbed
    
    # Ensure finiteness
    if not np.all(np.isfinite(perturbed)):
        perturbed = np.nan_to_num(perturbed, nan=mean_dist, posinf=mean_dist * 10, neginf=0.0)
    
    return perturbed
# EVOLVE-BLOCK-END
