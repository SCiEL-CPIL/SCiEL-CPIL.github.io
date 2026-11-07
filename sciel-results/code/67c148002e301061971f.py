"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Guided local search penalty: apply targeted perturbation to current tour edges.
    
    Strategy: Penalize ONLY the edges of the current local optimum tour, with a
    smooth, adaptive magnitude based on historical usage. This forces 2-opt to
    restructure the tour while keeping non-tour distances undistorted.
    """
    n = len(local_opt_tour)
    if n <= 2:
        return edge_distance.copy()
    
    # Work on a copy
    perturbed = edge_distance.copy()
    
    # Get the edges of the current tour (closed tour: tour[0]->tour[1]->...->tour[n-1]->tour[0])
    # For a tour represented as a permutation, edges are (tour[i], tour[(i+1)%n])
    tour = np.asarray(local_opt_tour)
    
    # Determine penalty scale - moderate enough to break the tour but not destroy structure
    # Use a base penalty that's a fraction of the average edge length
    avg_edge = np.mean(edge_distance)
    base_penalty = 0.3 * avg_edge
    
    # For each edge in the current tour, apply a smooth penalty based on usage
    for i in range(n):
        u = tour[i]
        v = tour[(i + 1) % n]
        usage = edge_n_used[u, v]
        
        # Smooth saturating function: penalty grows with usage but saturates
        # f(usage) = 1 - exp(-usage / scale) gives gentle start, strong for high usage
        # Scale usage so that ~5 uses gives substantial penalty
        penalty_factor = 1.0 - np.exp(-usage / 3.0)
        
        # Additional boost for edges in the CURRENT tour (always penalize at least somewhat)
        current_tour_boost = 0.5
        
        total_penalty = base_penalty * (current_tour_boost + penalty_factor)
        
        perturbed[u, v] += total_penalty
        perturbed[v, u] += total_penalty
    
    return perturbed
# EVOLVE-BLOCK-END
