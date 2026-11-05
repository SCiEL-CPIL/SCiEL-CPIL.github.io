"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def priority(adjacency):
    """Score vertices to maximize greedy independent set size.
    
    Strategy: combine negative degree with mean neighbor degree.
    This rewards picking low-degree vertices that are adjacent to high-degree hubs,
    effectively 'sacrificing' hubs early when forced, preserving more mid/low-degree
    vertices for the independent set.
    """
    A = np.asarray(adjacency, dtype=np.float64)
    
    # Degree of each vertex
    degrees = A.sum(axis=1)
    
    # Mean neighbor degree: for each vertex, average degree of its neighbors
    # (A @ degrees) / degrees, with safe division
    neighbor_degree_sum = A @ degrees
    mean_neighbor_degree = np.where(degrees > 0, neighbor_degree_sum / np.maximum(degrees, 1), 0.0)
    
    # Score: lower degree is better (negative), and higher mean neighbor degree is better
    # (we want to pick vertices that are connected to hubs, so hubs get "used up" early)
    # Normalize to avoid scale issues
    deg_norm = degrees / (degrees.max() + 1e-9)
    mnd_norm = mean_neighbor_degree / (mean_neighbor_degree.max() + 1e-9)
    
    # Combine: prioritize low degree, then high mean neighbor degree
    # Weight: degree matters more, but mean neighbor degree provides tie-breaking
    score = -deg_norm + 0.3 * mnd_norm
    
    return score
# EVOLVE-BLOCK-END
