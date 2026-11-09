"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def priority(adjacency):
    # Convert to numpy array if needed
    adj = np.asarray(adjacency, dtype=np.float64)
    n = adj.shape[0]
    
    # Compute degrees
    degrees = adj.sum(axis=1)
    
    # Compute 2-hop neighborhood size (number of vertices at distance 2)
    # This helps identify vertices in sparse regions
    # 2-hop count = (A^2)[i,j] summed over j, excluding direct neighbors
    A2 = adj @ adj
    # Subtract direct neighbors to get only distance-2 vertices
    two_hop_counts = A2.sum(axis=1) - degrees
    
    # Use a combination: prefer low degree and low 2-hop connectivity
    # This identifies vertices in sparse, isolated regions
    score = -degrees - 0.5 * two_hop_counts
    
    return score
# EVOLVE-BLOCK-END
