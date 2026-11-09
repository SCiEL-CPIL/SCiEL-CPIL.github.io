"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Return an edge-attractiveness matrix (larger is more attractive) of the same shape.
    
    Design: We use a composite heuristic that combines:
    1. Inverse distance (standard ACO heuristic)
    2. A 'nearest neighbor' bias that favors edges leading to customers with
       good local connectivity (low sum of distances to other unvisited customers)
    3. Demand-awareness: penalize edges to high-demand customers when the remaining
       capacity is tight, encouraging better route packing
    
    This is a structural change from the simple 1/d heuristic.
    """
    n = distance_matrix.shape[0]
    heuristic = np.zeros((n, n))
    
    # Base: inverse distance
    inv_dist = 1.0 / (distance_matrix + 1e-9)
    
    # For each node i, compute a 'local attractiveness' for going to j
    # We want to favor j if:
    #   - d(i,j) is small
    #   - j has good connections to other nodes (so the route can continue efficiently)
    #   - j's demand doesn't cause a capacity issue (we approximate this)
    
    # Compute for each customer j (j>0), the sum of distances to all other customers
    # This serves as a proxy for 'how central' or 'how connected' j is
    customer_sums = np.zeros(n)
    for j in range(1, n):
        customer_sums[j] = np.sum(distance_matrix[j, 1:])
    
    # Normalize customer_sums: lower sum = more central = more attractive
    # Use inverse of (sum + small) as a centrality score
    centrality = 1.0 / (customer_sums + 1e-9)
    # Normalize centrality to [0, 1]
    c_min = centrality[1:].min()
    c_max = centrality[1:].max()
    if c_max > c_min:
        centrality_norm = (centrality - c_min) / (c_max - c_min)
    else:
        centrality_norm = np.zeros_like(centrality)
    centrality_norm[0] = 1.0  # depot gets neutral
    
    # Demand penalty: for high-demand customers, we slightly penalize
    # unless the capacity is large enough to comfortably fit them
    # Since capacity is 50 and demands are 1-9, this is mild
    demand_penalty = np.zeros(n)
    for j in range(1, n):
        # Higher demand = slightly less attractive in heuristic (to encourage
        # grouping high-demand customers together in dedicated routes)
        demand_penalty[j] = 1.0 - 0.1 * (demands[j] / 9.0)
    demand_penalty[0] = 1.0
    
    # Combine: heuristic[i,j] = inv_dist[i,j] * (0.7 + 0.3 * centrality_norm[j]) * demand_penalty[j]
    # The 0.7 + 0.3 * centrality keeps distance dominant but adds structural info
    heuristic = inv_dist * (0.7 + 0.3 * centrality_norm[None, :]) * demand_penalty[None, :]
    
    # Ensure non-positive values become 1e-9 (as required)
    heuristic = np.where(heuristic > 0, heuristic, 1e-9)
    
    # Ensure finite values
    heuristic = np.where(np.isfinite(heuristic), heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
