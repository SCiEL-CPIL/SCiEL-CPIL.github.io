"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def priority(adjacency):
    n = adjacency.shape[0]
    degrees = adjacency.sum(axis=1).astype(float)
    # Sum of neighbor degrees for each vertex
    neighbor_deg_sum = adjacency @ degrees
    # Prefer low degree, and prefer vertices whose neighbors are high-degree
    # (excluding high-degree neighbors is "cheap")
    scores = -degrees + 0.5 * neighbor_deg_sum
    return scores
# EVOLVE-BLOCK-END
