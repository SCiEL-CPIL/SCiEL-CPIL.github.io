"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def priority(adjacency):
    """
    Compute priority scores for vertices.
    Higher score means the vertex is more likely to be selected earlier by the greedy decoder.
    The strategy is to prefer vertices that have a low degree (low cost to pick) 
    but whose neighbors have high degrees (high benefit of removing them).
    Score = mean_neighbor_degree - degree
    """
    # Ensure adjacency is a float matrix for matrix multiplication
    adj = np.asarray(adjacency, dtype=float)
    
    # Compute degrees
    degrees = adj.sum(axis=1)
    
    # Compute sum of degrees of neighbors
    # d_sum[i] = sum(degrees[j] for j in neighbors of i)
    neighbor_degree_sums = adj.dot(degrees)
    
    # Compute mean degree of neighbors
    # Handle zero degrees to avoid division by zero
    # If degree is 0, mean neighbor degree is 0
    with np.errstate(divide='ignore', invalid='ignore'):
        mean_neighbor_degrees = neighbor_degree_sums / degrees
        # Replace NaN (from 0/0) with 0
        mean_neighbor_degrees = np.nan_to_num(mean_neighbor_degrees, nan=0.0)
        
    # Score: We want to pick nodes that remove high-degree neighbors (high mean_neighbor_degree)
    # and have low own degree (low degree).
    scores = mean_neighbor_degrees - degrees
    
    return scores
# EVOLVE-BLOCK-END
