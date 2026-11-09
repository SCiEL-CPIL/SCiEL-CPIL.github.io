"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def priority(adjacency):
    # Convert input to numpy array for efficient operations
    adj = np.asarray(adjacency)
    
    # Calculate degree of each vertex
    degrees = adj.sum(axis=1)
    
    # Calculate sum of degrees of neighbors for each vertex
    # This serves as a tie-breaker: among vertices with the same degree,
    # we prefer those connected to higher-degree vertices (hubs),
    # as removing them clears more "blocking" capacity from the graph.
    neighbor_degree_sums = adj.dot(degrees)
    
    # Combine into a single score.
    # We scale the degree term heavily so that it acts as the primary key.
    # The neighbor degree sum acts as the secondary key for tie-breaking.
    # N=200, max neighbor sum ~ 200*199 ~ 40000.
    # Scale 100000 ensures degree difference of 1 outweighs any neighbor sum difference.
    scale = 100000.0
    
    # Lower degree -> Higher priority (less negative).
    # Higher neighbor sum -> Higher priority (more positive).
    scores = -degrees * scale + neighbor_degree_sums
    
    return scores.astype(float)
# EVOLVE-BLOCK-END
