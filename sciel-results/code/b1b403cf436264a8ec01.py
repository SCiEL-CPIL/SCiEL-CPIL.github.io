def priority(adjacency):
    n = adjacency.shape[0]
    deg = adjacency.sum(axis=1).astype(float)
    
    # Avoid division by zero
    safe_deg = deg + 1.0
    
    # Calculate sum of 1/deg for neighbors
    # inv_deg = 1 / safe_deg
    # neighbor_sum_inv_deg = adjacency @ inv_deg
    
    inv_deg = 1.0 / safe_deg
    neighbor_sum_inv_deg = adjacency @ inv_deg
    
    # Score:
    # Primary: Minimize degree
    # Secondary: Minimize sum of inverse degrees of neighbors (i.e., prefer neighbors with high degree)
    
    # Let's tune the weights.
    # Current: -10 * deg + mean_neighbor_deg
    # Mean neighbor deg = sum(deg[neighbors]) / deg
    # So current is approx -10 deg + (A @ deg) / deg
    
    # Proposed: -deg - neighbor_sum_inv_deg
    # Let's try scaling to match magnitudes.
    # deg is ~10-50.
    # neighbor_sum_inv_deg is ~ deg / avg_deg ~ 1-5.
    # So -deg dominates.
    
    # Maybe we should weight the neighbor term more?
    # Or use a different base score.
    
    # Let's try: Score = -deg - 2 * neighbor_sum_inv_deg
    scores = -deg - 2.0 * neighbor_sum_inv_deg
    
    return scores
