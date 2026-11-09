import numpy as np

def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    n = len(local_opt_tour)
    
    # Use median distance for robustness against outliers
    # Flatten to get all pairwise distances
    # edge_distance is symmetric, so we can just take upper triangle or all
    med_dist = np.median(edge_distance)
    
    # Normalize usage counts
    max_used = np.max(edge_n_used)
    if max_used == 0:
        # First iteration or no history
        norm_used = np.zeros_like(edge_n_used)
    else:
        norm_used = edge_n_used / max_used
        
    # Base penalty weight
    # We want the penalty for the most used edge to be significant, e.g., 50% of median distance
    # This forces 2-opt to consider alternatives
    base_penalty_weight = 0.5 * med_dist
    
    # Calculate historical penalty
    # Using a slight non-linearity: square the normalized usage to emphasize frequent edges
    historical_penalty = base_penalty_weight * (norm_used ** 1.5)
    
    # Create the new distance matrix
    penalized_dist = edge_distance.copy()
    penalized_dist += historical_penalty
    
    # Apply a strong penalty to the current tour edges to force change
    # This is crucial because 2-opt is a local search; if the current tour is still "locally attractive"
    # even after historical penalties, it might stay.
    # We add an extra penalty to ensure the current tour is no longer the local optimum.
    
    # Determine a strong penalty for current edges
    # Make it proportional to the median distance as well
    current_edge_penalty = 1.0 * med_dist
    
    for i in range(n):
        u = local_opt_tour[i]
        v = local_opt_tour[(i + 1) % n]
        penalized_dist[u, v] += current_edge_penalty
        penalized_dist[v, u] += current_edge_penalty
        
    return penalized_dist
