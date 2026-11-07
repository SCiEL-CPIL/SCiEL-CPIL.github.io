"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Guided local search penalty rule for TSP.
    
    Implements a hard edge-exclusion strategy: edges in the current local-optimal
    tour are assigned a very large finite distance, effectively excluding them
    from the subsequent 2-opt improvement. This forces the search to explore
    a distinct neighborhood by breaking the current tour structure.
    
    Args:
        edge_distance: True Euclidean distance matrix (n x n).
        local_opt_tour: Current tour as a list/array of node indices (closed tour).
        edge_n_used: Symmetric count matrix of how many times each edge has been used.
    
    Returns:
        A modified distance matrix of the same shape with finite values.
    """
    n = edge_distance.shape[0]
    modified = edge_distance.copy().astype(float)
    
    # Ensure finite values
    modified = np.where(np.isfinite(modified), modified, 0.0)
    
    # Extract edges from the current tour
    tour = np.asarray(local_opt_tour)
    
    # Handle potential edge cases
    if len(tour) < 2:
        return modified
    
    # Determine a large finite value for exclusion
    # We want these edges to be strictly worse than any valid edge.
    # Max possible distance in unit square is sqrt(2).
    # Using 100 * max_dist ensures they are far worse.
    max_dist = np.max(modified)
    if not np.isfinite(max_dist) or max_dist <= 0:
        max_dist = 1.0
    
    exclusion_value = 100.0 * max_dist
    
    # Apply exclusion to all edges in the current tour
    for i in range(len(tour)):
        u = tour[i]
        v = tour[(i + 1) % len(tour)]
        if u != v:
            modified[u, v] = exclusion_value
            modified[v, u] = exclusion_value
            
    # Optionally, add a small penalty to other frequently used edges to guide search
    # This is a standard GLS component, but the hard exclusion is the primary perturbation.
    if edge_n_used is not None and edge_n_used.size > 0:
        n_used = np.asarray(edge_n_used)
        # Only penalize edges NOT in the current tour (which are already excluded)
        # Create a mask for tour edges to zero out their penalty contribution
        tour_mask = np.zeros_like(n_used, dtype=bool)
        for i in range(len(tour)):
            u = tour[i]
            v = tour[(i + 1) % len(tour)]
            if u != v:
                tour_mask[u, v] = True
                tour_mask[v, u] = True
                
        # Soft penalty for other used edges
        # Scale penalty by count, but keep it small compared to exclusion
        max_used = np.max(n_used)
        if max_used > 0:
            # Penalty is proportional to usage count
            # We use a small factor so it doesn't interfere with the hard exclusion
            soft_penalty = 0.1 * max_dist * (n_used / max_used)
            # Zero out penalty for tour edges (already handled)
            soft_penalty[tour_mask] = 0.0
            modified += soft_penalty
            
    # Ensure all values are finite
    modified = np.where(np.isfinite(modified), modified, 0.0)
    
    # Ensure symmetry
    modified = (modified + modified.T) / 2.0
    
    return modified
# EVOLVE-BLOCK-END
