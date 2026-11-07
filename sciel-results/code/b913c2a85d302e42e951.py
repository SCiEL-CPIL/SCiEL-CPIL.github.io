import numpy as np

def heuristics(distance_matrix, coordinates, demands, capacity):
    d = distance_matrix.astype(float)
    n = d.shape[0]
    
    # Distance from each node to the depot (node 0)
    depot_dist = d[:, 0].copy()  # d[i, 0] for all i
    
    # Demand-aware structural heuristic:
    # 1 / (d[i,j] + beta * d[j,0] + gamma * demand[j] / capacity)
    # This biases against high-demand customers early, preserving capacity for more routes
    beta = 0.7
    gamma = 0.3
    
    # Build denominator: d[i,j] + beta * d[j,0] + gamma * (demand[j]/capacity)
    # d[j,0] is the same for all i, so we can broadcast
    demand_factor = (demands / capacity)[None, :]  # shape (1, n)
    denom = d + beta * depot_dist[None, :] + gamma * demand_factor
    
    # Add small epsilon to avoid division by zero
    epsilon = 1e-6
    heuristic = 1.0 / (denom + epsilon)
    
    # Set diagonal to 0
    np.fill_diagonal(heuristic, 0.0)
    
    # Replace non-positive or non-finite with 1e-9
    mask = ~np.isfinite(heuristic) | (heuristic <= 0)
    heuristic[mask] = 1e-9
    
    return heuristic
