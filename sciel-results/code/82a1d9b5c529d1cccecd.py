import numpy as np

def priority(adjacency):
    A = adjacency.astype(float)
    
    # Compute degrees
    degrees = np.sum(A, axis=1)
    
    # Compute weighted degrees: sum of degrees of neighbors
    # This is essentially A @ degrees
    weighted_degrees = A.dot(degrees)
    
    # Score: We want to pick vertices with low degree and low weighted degree.
    # The greedy decoder picks in descending score order.
    # So we assign negative values such that "good" candidates (low degree/weighted) have higher scores.
    # Combining them: Score = -(degrees + weighted_degrees)
    # We can weight them, e.g., degrees are more immediate.
    scores = -(degrees + 0.5 * weighted_degrees)
    
    return scores
