import numpy as np
import time

def heuristics(prizes, distance_matrix, max_len):
    """
    Compute heuristic matrix for ant colony orienteering.
    
    Combines local prize efficiency with cluster potential awareness.
    
    1. Prize Efficiency: (prize[j] / (d[i,j] + d[j,0]))^gamma
       Rewards nodes with high prize relative to the cost of visiting and returning.
       
    2. Distance Penalty: (1 / (1 + d[i,j]/max_len))^beta
       Penalizes moves that consume a large fraction of the remaining budget.
       
    3. Cluster Potential: (V_j / mean_V)^delta
       V_j is the prize-weighted sum of nearby high-value nodes.
       Rewards moving to nodes that are hubs of high-value clusters, 
       anticipating that future moves will also be high-value and cheap.
    """
    start_time = time.time()
    deadline = start_time + 280.0  # Safety margin

    n = len(prizes)
    prizes = np.asarray(prizes, dtype=float)
    distance_matrix = np.asarray(distance_matrix, dtype=float)
    
    eps = 1e-9
    
    # --- Step 1: Compute Cluster Potential for each node ---
    # For each node j, calculate the potential value of the local neighborhood.
    # This captures the "future opportunity cost" benefit of being in a dense cluster.
    
    k_neighbors = min(5, n - 1)
    decay_lambda = 2.0  # Controls how quickly influence decays with distance
    
    cluster_potential = np.zeros(n, dtype=float)
    
    for j in range(n):
        # Get distances from j to all other nodes
        dists = distance_matrix[j, :].copy()
        dists[j] = np.inf  # Exclude self
        
        # Find k nearest neighbors
        if n > 1:
            k_actual = min(k_neighbors, n - 1)
            # argpartition is faster than argsort for finding top-k
            nearest_indices = np.argpartition(dists, k_actual)[:k_actual]
            nearest_dists = dists[nearest_indices]
            nearest_prizes = prizes[nearest_indices]
            
            # Weighted sum of prizes, decaying with distance
            # We include node j's own prize in the potential as a base
            weights = np.exp(-decay_lambda * nearest_dists)
            neighbor_potential = np.sum(nearest_prizes * weights)
            
            # The potential of node j is its own prize plus the weighted prize of neighbors
            # This makes high-prize nodes in high-prize clusters very attractive
            cluster_potential[j] = prizes[j] + neighbor_potential
        else:
            cluster_potential[j] = prizes[j]
    
    # Normalize cluster potential to be around 1 on average
    mean_potential = np.mean(cluster_potential)
    if mean_potential < eps:
        mean_potential = eps
    normalized_potential = cluster_potential / mean_potential
    
    # --- Step 2: Compute Edge Heuristics ---
    
    # Parameters tuned to balance the three components
    gamma = 2.5  # Efficiency exponent
    beta = 1.5   # Distance penalty exponent
    delta = 0.5  # Cluster potential exponent
    
    # d[j,0] is the return cost from node j to start
    d_to_start = distance_matrix[:, 0].copy()  # shape (n,)
    
    # Effective cost: d[i,j] + d[j,0]
    eff_cost = distance_matrix + d_to_start[None, :]  # shape (n, n)
    eff_cost_safe = eff_cost + eps
    
    # Component 1: Prize Efficiency
    prize_row = prizes[None, :]  # shape (1, n)
    prize_eff = (prize_row / eff_cost_safe) ** gamma
    
    # Component 2: Distance Penalty
    dist_penalty = (1.0 / (1.0 + distance_matrix / max_len)) ** beta
    
    # Component 3: Cluster Potential
    # Broadcast normalized_potential to columns (potential of destination j)
    cluster_factor = (normalized_potential[None, :] + eps) ** delta
    
    # Combine components
    eta = prize_eff * dist_penalty * cluster_factor
    
    # Set eta[i,0] = 1e-9 for all i (no point visiting node 0)
    eta[:, 0] = 1e-9
    
    # Ensure all values are positive and finite
    eta = np.where(np.isfinite(eta) & (eta > 0), eta, 1e-9)
    
    # Check time
    if time.time() > deadline:
        # Fallback to simple efficiency if something went wrong or took too long
        d_j0 = distance_matrix[:, 0]
        costs = distance_matrix + d_j0[None, :] + eps
        fallback = (prizes[None, :] + eps) / costs
        fallback[:, 0] = 1e-9
        fallback = np.where(np.isfinite(fallback) & (fallback > 0), fallback, 1e-9)
        return fallback
    
    return eta
