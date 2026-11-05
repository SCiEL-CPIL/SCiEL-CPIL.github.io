import numpy as np

def priority(adjacency):
    n = adjacency.shape[0]
    if n == 0:
        return np.array([])
    
    # Convert to float for matrix operations if needed, but sum is enough
    degrees = adjacency.sum(axis=1).astype(float)
    
    # Calculate sum of neighbor degrees (weighted degree)
    # W[v] = sum(degrees[u] for u in neighbors(v))
    # This is equivalent to (adjacency @ degrees)
    weighted_degrees = adjacency.dot(degrees)
    
    # We want to pick vertices with low degree and low weighted degree.
    # Score should be high for "good" vertices (low constraint).
    # Let's use a negative sum or a negative product.
    # Negative sum is robust.
    # Score = -(degrees + weighted_degrees)
    
    # To make the scale less dependent on n, we can normalize, 
    # but the greedy decoder just sorts by score, so relative order matters.
    
    # Let's try: Score = - (degrees + 0.5 * weighted_degrees)
    # The 0.5 is a heuristic weight. Let's try 1.0 first as it captures 2-hop influence strongly.
    
    scores = -(degrees + weighted_degrees)
    
    return scores
