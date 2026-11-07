import numpy as np

def heuristics(distance_matrix, coordinates, demands, capacity):
    n = distance_matrix.shape[0]
    eps = 1e-9
    
    # 1. Base Heuristic: Inverse Distance
    base_heuristic = 1.0 / (distance_matrix + eps)
    
    # 2. Spatial Density Measure
    # For each node, find distance to k-th nearest neighbor
    k = 5
    # Sort distances for each row
    sorted_dists = np.sort(distance_matrix, axis=1)
    
    # Distance to k-th neighbor (index k, since index 0 is self)
    # If n < k+1, handle appropriately, but n=50 so it's fine
    kth_dists = sorted_dists[:, k]
    
    # Density is inverse of kth distance
    density = 1.0 / (kth_dists + eps)
    
    # Normalize density to [0, 1]
    d_min = np.min(density)
    d_max = np.max(density)
    if d_max > d_min:
        norm_density = (density - d_min) / (d_max - d_min)
    else:
        norm_density = np.ones_like(density)
        
    # 3. Combine Heuristics
    # Reward edges between nodes in dense areas
    # Factor = 0.5 + 0.5 * (norm_density[i] + norm_density[j]) / 2
    # This ranges from 0.5 (sparse) to 1.0 (dense)
    
    # Create matrix of average normalized densities
    # avg_density[i, j] = (norm_density[i] + norm_density[j]) / 2
    avg_density = (norm_density[:, None] + norm_density[None, :]) / 2.0
    
    # Density boost factor
    density_boost = 0.5 + 0.5 * avg_density
    
    # 4. Demand Penalty
    # Penalize edges to high demand nodes
    # Factor = 1 - (demand[j] / capacity)
    demand_factor = 1.0 - (demands / capacity)
    demand_factor = np.maximum(demand_factor, 0.0)
    
    # Combine
    # heuristic[i, j] = base[i, j] * density_boost[i, j] * demand_factor[j]
    heuristic = base_heuristic * density_boost * demand_factor[None, :]
    
    # Ensure positive
    heuristic = np.maximum(heuristic, 1e-9)
    
    return heuristic
