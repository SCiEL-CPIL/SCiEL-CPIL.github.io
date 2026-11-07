import numpy as np
import time

def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Computes an edge-attractiveness matrix using a density-weighted inverse distance heuristic.
    This structural change encourages the ACO to group customers in dense clusters,
    which is typically more efficient for CVRP than purely shortest-path edges.
    """
    start_time = time.time()
    deadline = start_time + 5.0  # 5 seconds safety margin for heuristic computation
    
    n = distance_matrix.shape[0]
    
    # Base heuristic: Inverse distance
    # Using power 1.0 (standard) as the trend from 2.0 -> 1.5 was improving.
    # The density term provides the "structural" improvement.
    eps = 1e-9
    d = distance_matrix + eps
    eta_base = 1.0 / d
    
    # Calculate local density for each node
    # Using a Gaussian kernel to smoothly measure local crowding
    sigma = 0.15
    # distance_matrix is NxN. 
    # density_j = sum_k exp(-d_jk / sigma)
    # We can do this efficiently with matrix operations
    density_matrix = np.exp(-distance_matrix / sigma)
    
    # Sum over neighbors (axis 1 sums over j for each i? No, we want density for each target j)
    # density_matrix[i, j] is the weight of i->j.
    # We want a vector density[j] = sum_i density_matrix[i, j]
    # Exclude the node itself? The diagonal is exp(0)=1. It's fine to include or exclude.
    # Let's exclude self to be precise about "neighbors".
    np.fill_diagonal(density_matrix, 0)
    
    local_density = np.sum(density_matrix, axis=0)
    
    # The depot (node 0) should probably have a neutral density or high?
    # The heuristic eta[i, j] is attractiveness of moving i->j.
    # So the weight should be associated with the TARGET node j.
    # density[j] represents how crowded j is.
    # High density -> High attractiveness? 
    # Yes, it's better to visit a node in a cluster than an isolated one, 
    # as it allows for short inter-visit distances.
    
    # Scale density to avoid dominating the distance term too much.
    # local_density ranges from 0 to ~N.
    # Let's normalize it to [0, 1] or similar.
    max_density = np.max(local_density) if np.max(local_density) > 0 else 1.0
    normalized_density = local_density / max_density
    
    # Combine: eta[i, j] = eta_base[i, j] * (1 + normalized_density[j])
    # We need to apply the density weight to the columns of eta_base.
    # eta_base is NxN. We want to multiply column j by (1 + normalized_density[j]).
    weight_vector = 1.0 + normalized_density
    
    eta = eta_base * weight_vector[np.newaxis, :]
    
    # Safety: Check deadline (though this is fast)
    if time.time() > deadline:
        # Fallback to simple inverse distance if something went wrong or took too long
        eta = 1.0 / (distance_matrix + eps)
        
    # Handle non-finite values
    eta = np.where(np.isfinite(eta) & (eta > 0), eta, 1e-9)
    
    return eta
