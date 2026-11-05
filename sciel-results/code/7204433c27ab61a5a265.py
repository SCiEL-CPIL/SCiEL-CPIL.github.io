import numpy as np

def priority(adjacency):
    # Ensure adjacency is a numpy array for fast operations
    adj = np.asarray(adjacency)
    
    # Compute degrees
    degrees = adj.sum(axis=1)
    
    # Compute sum of degrees of neighbors
    # This is equivalent to matrix-vector multiplication: adj @ degrees
    neighbor_degree_sums = adj.dot(degrees)
    
    # We want to prioritize vertices with low degree and low neighbor-degree-sum.
    # Using a weighted sum to make degree the primary factor.
    # If degrees are small (ER graphs with p~0.5, deg~100), 
    # a factor of N (200) or similar ensures degree dominates.
    # However, if degrees are very similar, the neighbor sum breaks ties effectively.
    
    # Let's use a score that strongly penalizes high degree, 
    # and moderately penalizes high neighbor degree sum.
    
    # Heuristic: Minimize (degree + neighbor_degree_sum)
    # To make degree the primary key:
    score = -(degrees * 1000.0 + neighbor_degree_sums)
    
    return score
