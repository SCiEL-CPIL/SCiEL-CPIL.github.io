"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, distance_matrix, max_len):
    """
    Computes the heuristic matrix using a Spatial Density and Additive Cost model.
    
    Unlike the multiplicative cost model, this approach uses an additive cost 
    (d[i,j] + d[j,0]) which is more standard for path length constraints. 
    Crucially, it replaces the point prize with a 'local density' score, which 
    is the sum of prizes of nearby nodes. This encourages the ant to visit 
    nodes that are in clusters of high-value targets, potentially allowing 
    multiple high-prize nodes to be collected in a short path.
    """
    prizes = np.asarray(prizes)
    distance_matrix = np.asarray(distance_matrix)
    
    n = prizes.shape[0]
    epsilon = 1e-9
    
    # 1. Calculate local prize density for each node
    # A node is attractive if it is surrounded by other high-prize nodes.
    # We use a fixed radius. For 50 points in unit square, 0.3 is a reasonable scale.
    radius = 0.3 
    density = np.zeros(n)
    for j in range(n):
        # Vectorized sum of prizes for nodes within radius of j
        mask = distance_matrix[j, :] < radius
        density[j] = np.sum(prizes[mask])
        
    # Ensure node 0 (start) has 0 prize/density impact if it's the target? 
    # The problem says prize 0 at node 0. So density[0] will just be sum of nearby prizes.
    # But we usually don't want to "visit" node 0 as a target in the middle.
    # The ACO loop handles "unvisited node j". Node 0 is start.
    
    # 2. Calculate additive cost: distance to go + distance to return
    # This represents the total "budget" consumed by adding node j to the current path end i.
    return_cost = distance_matrix[:, 0] # d[j, 0]
    additive_cost = distance_matrix + return_cost[None, :]
    
    # 3. Compute heuristic
    # High density and low additive cost -> High attractiveness
    heuristic = density[None, :] / (additive_cost + epsilon)
    
    # 4. Handle non-finite values and ensure positivity
    heuristic = np.where(np.isfinite(heuristic), heuristic, 1e-9)
    heuristic = np.maximum(heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
