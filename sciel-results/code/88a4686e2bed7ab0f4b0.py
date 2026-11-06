"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def priority(adjacency):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    n = adjacency.shape[0]
    degrees = adjacency.sum(axis=1).astype(float)
    
    # Calculate Clustering Coefficient for each vertex
    # C[i] = (2 * edges_in_Ni) / (deg_i * (deg_i - 1))
    clustering = np.zeros(n)
    
    # Optimization: Only compute for vertices with degree >= 2
    # Use a loop for clarity and sufficient speed for N=200
    for i in range(n):
        deg = int(degrees[i])
        if deg < 2:
            clustering[i] = 0.0
            continue
        
        # Get neighbors
        neighbors = np.where(adjacency[i] > 0)[0]
        
        # Count edges in the induced subgraph of neighbors
        # adjacency[np.ix_(neighbors, neighbors)] gives the subgraph
        # Summing it counts each edge twice
        edges_in_n = adjacency[np.ix_(neighbors, neighbors)].sum() / 2.0
        
        # Max possible edges in a graph of size 'deg'
        max_edges = deg * (deg - 1) / 2.0
        
        clustering[i] = edges_in_n / max_edges
        
        # Check time
        if time.time() > deadline:
            break

    # Score calculation
    # We want to prioritize vertices that are "cheap" to pick.
    # Cheapness is high if Degree is low.
    # Cheapness is also high if Clustering is high (neighbors are redundant).
    # Score = -Degree + Lambda * Clustering
    # Lambda = 20.0 allows clustering to override degree penalty in bridge cases.
    lambda_val = 20.0
    
    scores = -degrees + lambda_val * clustering
    
    return scores
# EVOLVE-BLOCK-END
