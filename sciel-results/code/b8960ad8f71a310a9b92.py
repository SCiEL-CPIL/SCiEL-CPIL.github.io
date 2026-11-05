"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def priority(adjacency):
    n = adjacency.shape[0]
    deg = adjacency.sum(axis=1).astype(float)
    
    # Compute sum of neighbor degrees for each vertex
    sum_neighbor_deg = adjacency.astype(float) @ deg
    
    # Mean neighbor degree, with safe division
    mean_neighbor_deg = np.zeros(n, dtype=float)
    mask = deg > 0
    mean_neighbor_deg[mask] = sum_neighbor_deg[mask] / deg[mask]
    
    # New term: Sum of inverse neighbor degrees
    # Vertices connected to low-degree vertices are penalized more
    # because blocking a low-degree vertex is "costlier" (it was a better candidate)
    inv_deg = 1.0 / (deg + 1.0)
    sum_inv_neighbor_deg = adjacency.astype(float) @ inv_deg
    
    # Score: -degree + beta * mean_neighbor_degree - gamma * sum_inv_neighbor_degree
    # Isolated vertices get a very high score so they're picked first
    beta = 0.5
    gamma = 0.5
    scores = -deg + beta * mean_neighbor_deg - gamma * sum_inv_neighbor_deg
    
    # Isolated vertices (degree 0) should be picked first
    scores[~mask] = 1e6
    
    return scores
# EVOLVE-BLOCK-END
