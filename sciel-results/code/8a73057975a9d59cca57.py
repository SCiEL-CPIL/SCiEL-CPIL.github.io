import numpy as np

def priority(adjacency):
    """
    Assigns a priority score to each vertex for the greedy MIS decoder.
    Higher score = picked earlier.
    
    Heuristic:
    A vertex is a good candidate for the independent set if:
    1. It has a low degree (it blocks few vertices).
    2. Its neighbors have high degrees (picking it removes high-degree 'blockers' from the graph, 
       preserving their other neighbors for future selection).
       
    Score = Average Degree of Neighbors - Own Degree
    """
    # Convert to float for matrix multiplication
    A = adjacency.astype(float)
    
    # Calculate degrees
    degrees = A.sum(axis=1)
    
    # Calculate sum of degrees of neighbors
    # sum_neigh_deg[i] = sum(degrees[j] for j in neighbors(i))
    sum_neigh_deg = A @ degrees
    
    # Calculate average degree of neighbors
    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        avg_neigh_deg = np.where(degrees > 0, sum_neigh_deg / degrees, 0.0)
        
    # Final Score
    # We want high avg_neigh_deg and low degrees
    score = avg_neigh_deg - degrees
    
    return score
