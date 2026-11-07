import numpy as np

def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    n = edge_distance.shape[0]
    
    # Calculate a scale factor based on the typical edge length
    # Mask out the diagonal to get mean of actual edges
    mask = ~np.eye(n, dtype=bool)
    if np.any(mask):
        avg_len = np.mean(edge_distance[mask])
    else:
        avg_len = 1.0
        
    # Ensure avg_len is not zero to avoid division issues or zero penalty
    if avg_len < 1e-10:
        avg_len = 1.0
        
    # Define the penalty strength
    # We want the penalty to be a fraction of the edge length per unit of usage.
    # If an edge is used 10 times, we want to add ~50% of its length?
    # Let's try adding 5% of the average edge length per usage count.
    # So if used 10 times, penalty is 0.5 * avg_len.
    # This makes used edges significantly more expensive.
    penalty_factor = 0.05 * avg_len
    
    # Create the modified distance matrix
    modified_dist = edge_distance.copy()
    
    # Add the penalty
    # We only care about off-diagonal elements for the TSP tour
    # But adding to diagonal is harmless as long as it doesn't become negative
    # and 2-opt doesn't pick self-loops.
    modified_dist += penalty_factor * edge_n_used
    
    # Ensure the matrix is symmetric (it should be, but just in case of floating point errors)
    # modified_dist = (modified_dist + modified_dist.T) / 2.0
    
    # Ensure diagonal is 0 or a large value? 
    # Usually 0. If we add penalty, it becomes positive. 
    # 2-opt implementations typically ignore diagonal or handle it.
    # To be safe, we can reset diagonal to 0 if it was 0, or leave it.
    # The prompt says "finite matrix". 
    # Let's keep the penalty on diagonal too, it won't affect 2-opt swaps which involve 4 distinct nodes.
    
    return modified_dist
