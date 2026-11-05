"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def priority(adjacency):
    """
    Computes priority scores for vertices to maximize Independent Set size.
    Strategy: Prioritize vertices that are connected to high-degree nodes (hubs)
    but have low degree themselves. This effectively 'prunes' hubs from the
    candidate pool by selecting their low-degree neighbors first, which is
    more efficient than selecting the hubs themselves.
    """
    # Convert to numpy array for efficient operations
    adj = np.asarray(adjacency, dtype=np.float64)
    
    # Calculate degrees
    degrees = adj.sum(axis=1)
    
    # Calculate sum of degrees of neighbors
    neighbor_degree_sum = adj @ degrees
    
    # Calculate mean degree of neighbors
    # Add 1 to denominator to avoid division by zero and smooth the score
    mean_neighbor_degree = neighbor_degree_sum / (degrees + 1)
    
    # Score: Reward high neighbor degree (connected to hubs)
    # Penalize own degree (avoid picking hubs)
    # This aligns with the finding that rewarding vertices connected to
    # high-degree nodes improves the independent set size.
    score = mean_neighbor_degree - degrees
    
    return score
# EVOLVE-BLOCK-END
