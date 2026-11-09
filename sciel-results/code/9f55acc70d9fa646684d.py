"""Guided local search penalty for TSP; Global count-proportional multiplicative penalty."""
# EVOLVE-BLOCK-START
import numpy as np

def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Applies a global multiplicative penalty to all edges based on their historical usage count.
    This modifies the entire distance matrix to discourage edges that have been used in previous
    local optima, helping 2-opt explore new regions of the search space.
    
    Args:
    edge_distance: np.ndarray, true distance matrix.
    local_opt_tour: list or np.ndarray, the current tour as a sequence of node indices.
    edge_n_used: np.ndarray, symmetric matrix of usage counts.
    
    Returns:
    np.ndarray: Modified distance matrix for the next 2-opt phase.
    """
    # Copy the true distances
    new_dist = edge_distance.copy()
    
    # Apply penalty to all edges based on edge_n_used
    # The penalty factor is 1 + alpha * count
    # We use a small alpha to avoid distorting the geometry too much,
    # but large enough to influence 2-opt decisions.
    # Since edge_n_used accumulates over 20 rounds, counts can be up to ~20.
    # A factor of 1 + 0.1 * count means an edge used 10 times gets 2x distance.
    # This is a gentle push away from previously used edges globally.
    
    # Ensure edge_n_used is float for safe arithmetic
    counts = edge_n_used.astype(float)
    
    # Calculate penalty factor: 1 + 0.1 * count
    # Cap the count influence to avoid extreme values if counts get very high (though 20 rounds is low)
    # Max count is likely < 20, so 1 + 0.1*20 = 3.0 max factor.
    factor = 1.0 + 0.1 * counts
    
    # Apply the factor to the distance matrix
    # This is an element-wise multiplication
    new_dist = new_dist * factor
    
    # Ensure the matrix remains symmetric (it should be, but just in case of floating point issues)
    new_dist = (new_dist + new_dist.T) / 2.0
    
    # Ensure finite values (should be, but safety check)
    if not np.all(np.isfinite(new_dist)):
        return edge_distance.copy()
        
    return new_dist
# EVOLVE-BLOCK-END
