"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def priority(adjacency):
    """
    Calculates a priority score for each vertex to maximize the size of the
    greedy independent set.
    
    Heuristic:
    1. Prefer vertices with lower degree (fewer neighbors to block).
    2. Tie-breaker: Prefer vertices whose neighbors have higher degrees / lower "value".
       Rationale: Picking a vertex 'v' blocks its neighbors. 
       The "value" of a vertex u is inversely proportional to its degree 
       (low degree vertices are 'easy wins' and blocking them is costly).
       Therefore, we prefer to block high-degree neighbors (low value) 
       over low-degree neighbors (high value).
       We approximate this by minimizing the sum of inverse degrees of neighbors.
    """
    start_time = time.time()
    
    # Convert to integer array for efficient arithmetic
    adj = np.asarray(adjacency, dtype=np.int8)
    n = adj.shape[0]
    
    # Calculate degrees
    degrees = adj.sum(axis=1).astype(np.float64)
    
    # Calculate sum of inverse degrees of neighbors for each vertex
    # To avoid division by zero, we use (deg + 1)
    # inv_deg = 1.0 / (degrees + 1.0)
    # neighbor_inv_degree_sums = adj @ inv_deg
    # Note: adj is 0/1. adj @ inv_deg gives sum of inv_deg of neighbors.
    
    inv_deg = 1.0 / (degrees + 1.0)
    neighbor_inv_degree_sums = (adj @ inv_deg).astype(np.float64)
    
    # Second-order: sum of degrees of neighbors (how many other vertices each neighbor blocks)
    # Picking v blocks its neighbors; if those neighbors have high degree, they would have
    # blocked many other vertices. So we prefer vertices whose neighbors have HIGH total degree.
    neighbor_degree_sums = (adj @ degrees).astype(np.float64)
    
    # Define a constant M large enough to make degree the primary factor
    # Max degree ~ 200. Max inv sum ~ 200 * 1 = 200.
    # Max neighbor_degree_sums ~ 200 * 200 = 40000.
    # We want degree to dominate.
    M = 1000.0
    M2 = 1.0  # weight for neighbor_degree_sums (normalized by dividing by n)
    
    # Normalize neighbor_degree_sums to be comparable in scale
    neighbor_degree_sums_norm = neighbor_degree_sums / n
    
    # We want to MINIMIZE degree, MINIMIZE neighbor_inv_degree_sums, and MAXIMIZE neighbor_degree_sums_norm.
    # So: cost = degree * M + neighbor_inv_degree_sums - neighbor_degree_sums_norm * M2
    costs = degrees * M + neighbor_inv_degree_sums - neighbor_degree_sums_norm * M2
    priorities = -costs
    
    # Deadline guard (unlikely to trigger for N=200, but safe)
    if time.time() - start_time > 110:
         return priorities.astype(float)
         
    return priorities.astype(float)
# EVOLVE-BLOCK-END
