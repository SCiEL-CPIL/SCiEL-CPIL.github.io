"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def priority(adjacency):
    """
    Assigns a priority score to each vertex for the greedy independent set decoder.
    Higher scores are picked first.
    
    Strategy: 
    We model the "cost" of selecting a vertex v as the sum of its own degree and the 
    "value" of its neighbors. 
    We define the value of a neighbor u as inversely proportional to its degree (1/(deg(u)+1)).
    Low-degree neighbors are "valuable" candidates for the independent set, so blocking them 
    (by picking v) is costly. High-degree neighbors (hubs) are "cheap" to block because 
    they would have blocked many others anyway.
    
    Thus, we prefer vertices v that:
    1. Have low degree (low direct cost).
    2. Are connected to high-degree neighbors (low cost to block valuable nodes).
    
    This refines the "Sum of Neighbor Degrees" heuristic by using a harmonic measure 
    of neighbor quality, which is more robust to the specific distribution of degrees.
    """
    start_time = time.time()
    
    # Convert to numpy array
    A = np.asarray(adjacency, dtype=float)
    N = A.shape[0]
    
    if N == 0:
        return np.array([])
        
    # 1. Compute degrees
    deg = A.sum(axis=1)
    
    # 2. Compute Inverse Degree of each vertex
    # Adding 1 to avoid division by zero for isolated nodes
    inv_deg = 1.0 / (deg + 1.0)
    
    # 3. Compute Sum of Inverse Degrees of Neighbors
    # For each vertex v, this is sum_{u in N(v)} 1/(deg(u)+1)
    # This represents the "value" of the neighbors we are about to block.
    # Lower value is better (we are blocking "cheap" hubs).
    neighbor_value_sum = A @ inv_deg
    
    # 4. Calculate Raw Cost
    # Cost = Own Degree + Weight * Neighbor Value Sum
    # We want to minimize this cost.
    # We use a weight of 1.0 for both terms to balance direct degree cost 
    # against the opportunity cost of blocking neighbors.
    beta = 1.0
    raw_cost = deg + beta * neighbor_value_sum
    
    # 5. Normalize Cost to [0, 1]
    # Normalization ensures that if the graph density changes significantly,
    # the relative ordering remains stable and not dominated by one feature's scale.
    min_cost = raw_cost.min()
    max_cost = raw_cost.max()
    range_cost = max_cost - min_cost
    
    if range_cost > 0:
        norm_cost = (raw_cost - min_cost) / range_cost
    else:
        norm_cost = np.zeros_like(raw_cost)
        
    # 6. Score is negative normalized cost
    # Higher score = Lower cost = Better choice
    score = -norm_cost
    
    # Safety check for time
    if time.time() - start_time > 10.0:
        # Fallback to simple degree-based score if something went wrong
        return -deg.astype(float)
        
    return score
# EVOLVE-BLOCK-END
