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
    2. Secondary: Prefer vertices whose neighbors have higher degrees (sum).
       Rationale: Picking a vertex 'v' blocks its neighbors. If the blocked
       neighbors have high degrees, it is beneficial because those high-degree
       vertices would have blocked many other vertices if they had been picked.
    3. Tertiary: Prefer vertices with higher "effective degree reduction".
       We approximate this by looking at the sum of squares of neighbor degrees.
       Higher variance in neighbor degrees suggests a more heterogeneous neighborhood,
       which might be better to clear out early.
    4. Quaternary: Tie-breaker using inverse degree of neighbors.
       Rationale: Among vertices with similar degree and neighbor-degree-sum,
       prefer those whose neighbors have high degrees (low inverse degree).
    """
    start_time = time.time()
    
    # Convert to integer array for efficient arithmetic
    adj = np.asarray(adjacency, dtype=np.int8)
    n = adj.shape[0]
    
    # Calculate degrees
    degrees = adj.sum(axis=1).astype(np.float64)
    
    # Calculate sum of degrees of neighbors for each vertex
    neighbor_degree_sums = (adj @ degrees).astype(np.float64)
    
    # Calculate sum of squared degrees of neighbors for each vertex
    neighbor_degree_sq_sums = (adj @ (degrees ** 2)).astype(np.float64)
    
    # Calculate sum of inverse degrees of neighbors for each vertex
    inv_deg = 1.0 / (degrees + 1.0)
    neighbor_inv_degree_sums = (adj @ inv_deg).astype(np.float64)
    
    # Define weights for the cost function.
    # We want to MINIMIZE: degree, neighbor_inv_degree_sums
    # and MAXIMIZE: neighbor_degree_sums, neighbor_degree_sq_sums
    
    # Scale analysis for N=200:
    # degree: [0, 199]
    # neighbor_degree_sums: [0, ~40000]
    # neighbor_degree_sq_sums: [0, ~40000*200 = 8,000,000] -> too large, need scaling
    # neighbor_inv_degree_sums: [0, 200]
    
    # We want degree to be the primary factor.
    # Let's define cost = M * degree - w1 * neighbor_degree_sums - w2 * neighbor_degree_sq_sums + w3 * neighbor_inv_degree_sums
    # We want M * degree to dominate.
    # Max neighbor_degree_sums is ~40000.
    # Max neighbor_degree_sq_sums is ~8,000,000. This is much larger.
    # We should normalize neighbor_degree_sq_sums or use a very small weight.
    # Alternatively, we can just use neighbor_degree_sums and drop the sq term if it's too noisy.
    # Or, we can compute "mean neighbor degree" which is bounded by N.
    # mean_neighbor_deg = neighbor_degree_sums / (degrees + 1)
    # This is bounded by ~200. This is a better secondary feature.
    
    mean_neighbor_deg = neighbor_degree_sums / (degrees + 1.0)
    
    # Now:
    # degree: [0, 199]
    # mean_neighbor_deg: [0, 200]
    # neighbor_inv_degree_sums: [0, 200]
    
    # Let's use:
    # cost = M * degree - w1 * mean_neighbor_deg + w2 * neighbor_inv_degree_sums
    # We want degree to dominate.
    # If M = 1000, then a change of 1 in degree is 1000.
    # Max change in mean_neighbor_deg is 200. So w1 should be < 1000/200 = 5.
    # Let's set w1 = 1.0. Then max contribution is 200.
    # Max change in neighbor_inv_degree_sums is 200.
    # Let's set w2 = 1.0. Then max contribution is 200.
    # Total secondary contribution max is 400.
    # So M = 1000 is safe.
    
    M = 1000.0
    w1 = 1.0
    w2 = 1.0
    
    costs = M * degrees - w1 * mean_neighbor_deg + w2 * neighbor_inv_degree_sums
    priorities = -costs
    
    # Deadline guard
    if time.time() - start_time > 110:
        return priorities.astype(float)
        
    return priorities.astype(float)
# EVOLVE-BLOCK-END
