"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def priority(adjacency):
    n = adjacency.shape[0]
    deg = adjacency.sum(axis=1).astype(float)
    
    # Handle isolated vertices: they should be picked first (highest score)
    # For non-isolated: score = -deg + beta * mean_neighbor_degree
    # This prefers low-degree vertices, and among those, those adjacent to high-degree vertices
    
    # Compute sum of neighbor degrees for each vertex
    # sum_neighbor_deg[v] = sum of deg[u] for u in N(v)
    # This is A @ deg
    sum_neighbor_deg = adjacency.astype(float) @ deg
    
    # Mean neighbor degree, with safe division
    mean_neighbor_deg = np.zeros(n, dtype=float)
    mask = deg > 0
    mean_neighbor_deg[mask] = sum_neighbor_deg[mask] / deg[mask]
    
    # Score: -degree + beta * mean_neighbor_degree
    # Isolated vertices get a very high score so they're picked first
    beta = 0.5
    scores = -deg + beta * mean_neighbor_deg
    
    # Isolated vertices (degree 0) should be picked first
    scores[~mask] = 1e6
    
    return scores
# EVOLVE-BLOCK-END
