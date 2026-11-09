"""
Savings-based heuristic for ACO CVRP: uses Clarke-Wright savings
to identify high-value edges for route construction, enhanced with
demand-efficiency weighting.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time


def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Compute an edge-attractiveness matrix (larger = more attractive) for ACO CVRP.
    
    Approach:
      1. Compute Clarke-Wright savings for all customer-customer pairs.
      2. For depot edges, use inverse distance.
      3. Boost customer-customer edges based on demand efficiency (grouping heavy loads).
      4. Ensure positive values and finite results.
    
    Returns an n x n matrix of the same shape as distance_matrix.
    """
    start_time = time.time()
    deadline = start_time + 280.0

    n = distance_matrix.shape[0]
    if distance_matrix.shape[1] != n:
        return np.full((n, n), 1e-9)

    demands = np.asarray(demands, dtype=np.float64)
    if demands.shape[0] != n:
        demands = demands[:n]

    # Safe distance matrix to avoid division by zero
    dist_safe = np.maximum(distance_matrix, 1e-9)
    
    # Initialize heuristic with inverse distance (standard)
    heuristic = 1.0 / dist_safe
    
    # Compute savings for customer-customer edges
    # Savings S_ij = d_0i + d_0j - d_ij
    # We only apply this to i, j != 0
    
    if n > 1:
        # Extract depot row/col
        d0 = distance_matrix[0, :]
        
        # Compute savings matrix for all pairs
        # S_ij = d0[i] + d0[j] - d_ij
        savings = d0[:, None] + d0[None, :] - distance_matrix
        
        # Only consider positive savings (triangle inequality usually holds, but just in case)
        savings = np.maximum(savings, 0.0)
        
        # Mask out depot edges (row 0 and col 0) because savings logic doesn't apply directly
        # For depot edges, we keep the inverse distance
        depot_mask = np.zeros((n, n), dtype=bool)
        depot_mask[0, :] = True
        depot_mask[:, 0] = True
        
        # For customer-customer edges, combine savings with demand efficiency
        # Demand efficiency: higher if demands[i] + demands[j] is a larger fraction of capacity
        # This encourages grouping customers that fill the truck well
        demand_sum = demands[:, None] + demands[None, :]
        efficiency = 1.0 + 0.5 * (demand_sum / capacity)
        
        # The savings value is in distance units. Inverse distance is 1/distance.
        # To make them comparable, we might need to scale savings.
        # Let's use the savings directly as a multiplier or add it?
        # If we use savings directly, large distances might dominate.
        # Let's try: heuristic_cc = savings * efficiency
        # And for depot: heuristic_dep = 1/distance
        
        # However, savings can be 0 or negative (if triangle inequality is violated or d_ij is large).
        # We want to preserve the "short distance" preference too.
        # Let's blend: heuristic_cc = (1/d_ij) * efficiency * (1 + savings_scale * savings)
        # Or simpler: just use savings as the primary signal for CC edges, as it captures the "merging" benefit.
        
        # Let's try a weighted sum:
        # Base attractiveness: 1/d_ij
        # Savings boost: savings / max_savings (normalized)
        
        max_savings = np.max(savings[~depot_mask]) if np.any(~depot_mask) else 1.0
        if max_savings < 1e-9:
            max_savings = 1.0
            
        norm_savings = savings / max_savings
        
        # Combine: 
        # For CC edges: 1/d_ij * efficiency * (1 + norm_savings)
        # This ensures short edges are preferred, but edges with high savings (good clustering) are boosted further.
        
        cc_boost = efficiency * (1.0 + norm_savings)
        
        # Apply to non-depot edges
        heuristic[~depot_mask] = (1.0 / dist_safe[~depot_mask]) * cc_boost[~depot_mask]
        
        # Depot edges remain 1/d_ij

    # Ensure all entries are strictly positive
    heuristic = np.maximum(heuristic, 1e-9)

    # Final check: no NaN or Inf
    if not np.all(np.isfinite(heuristic)):
        heuristic = np.where(np.isfinite(heuristic), heuristic, 1e-9)
        heuristic = np.maximum(heuristic, 1e-9)

    return heuristic
# EVOLVE-BLOCK-END
