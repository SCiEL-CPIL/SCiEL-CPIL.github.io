"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def priority(adjacency):
    """
    Score each vertex for greedy MIS selection.
    
    Uses a combination of:
    - Negative degree (prefer low-degree vertices)
    - Positive mean neighbor degree (prefer vertices that block high-degree neighbors)
    
    The greedy decoder picks vertices in descending score order.
    """
    adjacency = np.asarray(adjacency, dtype=np.float64)
    n = adjacency.shape[0]
    
    # Degree of each vertex
    degrees = adjacency.sum(axis=1)
    
    # Mean degree of neighbors for each vertex
    # Sum of degrees of neighbors = adjacency @ degrees
    neighbor_degree_sums = adjacency @ degrees
    
    # Avoid division by zero for isolated vertices
    safe_degrees = np.where(degrees > 0, degrees, 1.0)
    mean_neighbor_degree = neighbor_degree_sums / safe_degrees
    
    # For isolated vertices (degree 0), mean_neighbor_degree is 0
    # Score: prefer low degree, reward high mean neighbor degree
    # Beta controls the weight of the neighbor degree term
    beta = 1.0
    scores = -degrees + beta * mean_neighbor_degree
    
    return scores
# EVOLVE-BLOCK-END
