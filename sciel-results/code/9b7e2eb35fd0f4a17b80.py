import numpy as np
import time

def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Global Additive Penalty with Top-K Exclusion for Guided Local Search.
    
    Strategy:
    1. Apply a global additive penalty to ALL edges based on their usage count in edge_n_used.
       This is the standard GLS approach: penalize edges that have been part of local optima 
       in previous iterations to encourage the search to explore different regions of the 
       solution space.
    2. Additionally, identify the top-K most frequently used edges and set their cost to a 
       very large value (effectively excluding them). This forces the 2-opt to find 
       completely different edges for the most persistent "bad" edges, promoting a 
       structural break rather than just a local adjustment.
    3. The penalty magnitude is scaled by the maximum usage count to ensure penalties are 
       significant relative to the edge lengths.
    
    This differs from the current multiplicative approach by:
    - Penalizing ALL edges, not just those in the current tour, which provides a broader 
      landscape distortion.
    - Using additive penalties, which are more standard in GLS and can be more effective 
      in pushing the search away from local optima.
    - Hard-excluding the most problematic edges to ensure a significant change in the tour.
    """
    # Safety check
    if edge_n_used.size == 0:
        return edge_distance.copy()
    
    # Create a copy of the distance matrix to modify
    modified_distance = edge_distance.copy()
    
    # Get the number of nodes
    n_nodes = edge_distance.shape[0]
    
    # Calculate the maximum usage count for normalization
    max_used = np.max(edge_n_used)
    if max_used == 0:
        return modified_distance
    
    # Calculate the average edge length for scaling the penalty
    # We use the true distances from edge_distance
    # To avoid O(N^2) computation every time, we can estimate or use a fixed scale
    # Since points are in unit square, typical edge lengths are ~0.1 to 1.0
    # Let's use a robust scale: mean of all non-zero distances? No, that's expensive.
    # Let's use a fixed base penalty that is a fraction of the typical edge length.
    # Typical edge length in 100 points in unit square is approx 0.1-0.2 for nearest neighbors, 
    # but tour edges can be longer.
    # Let's estimate the scale from the current tour's edges to be adaptive but cheap.
    
    # Extract edges from the current tour to estimate scale
    tour_edges = []
    tour_len = len(local_opt_tour)
    if tour_len > 1:
        for i in range(tour_len - 1):
            tour_edges.append((local_opt_tour[i], local_opt_tour[i + 1]))
        tour_edges.append((local_opt_tour[-1], local_opt_tour[0]))
        
        if tour_edges:
            lengths = np.array([edge_distance[u, v] for u, v in tour_edges])
            mean_len = np.mean(lengths)
        else:
            mean_len = 0.5
    else:
        mean_len = 0.5
    
    # Base penalty per usage unit
    # We want the penalty to be significant. 
    # If an edge is used max_used times, it should get a penalty of ~ base_penalty * max_used
    # Let's set base_penalty such that max_used * base_penalty is a significant fraction of mean_len
    # e.g., 0.5 * mean_len
    base_penalty = (0.5 * mean_len) / max_used
    
    # Apply global additive penalty
    # penalty[i, j] = base_penalty * edge_n_used[i, j]
    penalty_matrix = base_penalty * edge_n_used
    
    # Add penalty to the distance matrix
    modified_distance += penalty_matrix
    
    # Hard exclusion of top-K edges
    # Find the top-K most used edges
    # We want to exclude edges that are very likely to be part of the local optimum structure.
    # Let's pick the top 5% of edges by usage count, or a fixed number like 10-20 edges.
    # For 100 nodes, 10-20 edges is a small fraction.
    
    # Flatten the usage matrix to find top edges
    # Since it's symmetric, we only need to consider i < j
    i_idx, j_idx = np.triu_indices(n_nodes, k=1)
    usage_counts = edge_n_used[i_idx, j_idx]
    
    # Find the top K indices
    # Let's try K = 10
    K = 10
    if len(usage_counts) > K:
        top_k_indices = np.argpartition(usage_counts, -K)[-K:]
        top_k_i = i_idx[top_k_indices]
        top_k_j = j_idx[top_k_indices]
        
        # Set these edges to a very large value
        # Use a value larger than any possible tour length
        # Max tour length is roughly 100 * 1.4 (diagonal of square) ~ 140
        # So 1000 is safe
        large_value = 1000.0
        
        modified_distance[top_k_i, top_k_j] = large_value
        modified_distance[top_k_j, top_k_i] = large_value
    
    # Ensure the matrix is symmetric and finite
    # The penalty matrix is symmetric, so modified_distance should be symmetric.
    # But just in case, enforce symmetry
    modified_distance = (modified_distance + modified_distance.T) / 2.0
    
    # Check for any non-finite values
    if not np.all(np.isfinite(modified_distance)):
        # Fallback to original if something went wrong
        return edge_distance.copy()
        
    return modified_distance
