"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def priority(adjacency):
    """
    Calculates a priority score for each vertex to maximize the size of the
    greedy independent set.
    
    Heuristic:
    1. Primary: Prefer vertices with lower degree (fewer neighbors to block).
    2. Secondary: Prefer vertices whose neighbors have higher average degrees.
       Rationale: Picking a vertex 'v' blocks its neighbors. If the blocked
       neighbors are high-degree "hubs", it is beneficial because those hubs
       would have blocked many other vertices if they had been picked.
       By blocking them early, we 'save' their neighbors from being blocked.
       This is captured by maximizing the mean degree of neighbors.
       
    The score is defined as:
    Priority(v) = -degree(v) + mean_neighbor_degree(v)
    
    This balances the immediate cost (degree) against the future benefit (savings from blocking hubs).
    """
    start_time = time.time()
    
    # Convert to integer array for efficient arithmetic
    adj = np.asarray(adjacency, dtype=np.int8)
    n = adj.shape[0]
    
    # Calculate degrees
    degrees = adj.sum(axis=1).astype(np.float64)
    
    # Calculate sum of degrees of neighbors for each vertex
    neighbor_degree_sums = (adj @ degrees).astype(np.float64)
    
    # Calculate mean degree of neighbors
    # Add 1.0 to denominator to avoid division by zero and smooth the term
    mean_neighbor_deg = neighbor_degree_sums / (degrees + 1.0)
    
    # Define the priority score
    # We want to MINIMIZE degree and MAXIMIZE mean_neighbor_deg
    # So we subtract degree and add mean_neighbor_deg
    
    # Weights:
    # degree range [0, 199]
    # mean_neighbor_deg range [0, ~200]
    # Using 1:1 weighting allows high-quality neighbors to offset a slightly higher degree.
    # e.g., a degree 2 vertex with mean neighbor degree 50 (score 48) beats 
    # a degree 1 vertex with mean neighbor degree 1 (score 0).
    
    priorities = -degrees + mean_neighbor_deg
    
    # Deadline guard
    if time.time() - start_time > 110:
        return priorities.astype(float)
        
    return priorities.astype(float)
# EVOLVE-BLOCK-END
