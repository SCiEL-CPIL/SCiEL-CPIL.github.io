import numpy as np

def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Perturbation step for Guided Local Search.
    Adds an additive penalty to edge distances proportional to their usage count.
    The penalty is scaled by the median edge length to be instance-adaptive.
    """
    # Calculate a scale factor based on the typical edge length
    # Using median is robust to outliers (long edges crossing the square)
    # We exclude zeros if we consider only off-diagonal, but median of whole matrix 
    # for 100x100 is dominated by off-diagonals since diagonal is 0 (100 zeros vs 9900 non-zeros).
    # Actually, 100 zeros in 10000 elements is 1%. Median will be a non-zero value.
    
    # To be strictly safe and fast, we can just use a percentile or median of non-zero values.
    # But np.median on the whole matrix is fine and fast.
    median_len = np.median(edge_distance)
    
    # If median is 0 (all points identical?), return original
    if median_len == 0:
        return edge_distance.copy()
        
    # Penalty coefficient: 5% of median length per use
    # After 20 uses, penalty is 100% of median length.
    # This should be strong enough to disrupt local optima.
    alpha = 0.05
    
    # Additive penalty
    # edge_n_used is a count matrix.
    # We add alpha * median_len * count to the true distance.
    penalty = alpha * median_len * edge_n_used
    
    # Return the modified distance matrix
    return edge_distance + penalty
