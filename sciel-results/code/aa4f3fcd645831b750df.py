import numpy as np

def heuristics(prizes, distance_matrix, max_len):
    """
    Computes a heuristic matrix for Ant Colony Optimization on the Orienteering Problem.
    
    Structural Change: Uses a prize-weighted local centrality (cluster value) 
    to identify nodes that are hubs of high-value regions, rather than just 
    relying on individual prize and distance.
    
    Formula: eta[i, j] = (prize[j] * ClusterValue[j]) / (distance[i, j] + epsilon)
    Where ClusterValue[j] = sum(k != j) prize[k] / (distance[j, k] + epsilon)
    """
    prizes = np.asarray(prizes, dtype=np.float64)
    distance_matrix = np.asarray(distance_matrix, dtype=np.float64)
    
    n = len(prizes)
    epsilon = 1e-9
    
    # Compute Cluster Value (Prize-Weighted Inverse Distance Sum) for each node j
    # This represents the "potential" of node j to access other prizes.
    # D_inv is (n, n) where D_inv[j, k] = 1 / (d[j, k] + eps)
    # Set diagonal to 0 to avoid self-influence
    D_inv = 1.0 / (distance_matrix + epsilon)
    np.fill_diagonal(D_inv, 0.0)
    
    # ClusterValue[j] = sum_k prizes[k] * D_inv[j, k]
    # Shape: (n,)
    cluster_values = D_inv @ prizes
    
    # Construct the heuristic matrix
    # eta[i, j] = (prizes[j] * cluster_values[j]) / (distance[i, j] + epsilon)
    # Numerator: (n, n) where each row i has the same vector prizes * cluster_values
    # Denominator: (n, n) distance matrix
    
    # We want to favor moves to j that have high own prize AND high cluster value, 
    # and are close to i.
    
    # Note: We do NOT include distance to origin (d[j, 0]) in the denominator here,
    # as the feasibility check in the ACO loop handles the hard constraint, and 
    # including it in the heuristic often biases the search away from valid but 
    # far-from-origin nodes prematurely. The "Cluster Value" implicitly captures 
    # global structure.
    
    numerator = prizes[None, :] * cluster_values[None, :]
    denominator = distance_matrix + epsilon
    
    heur = numerator / denominator
    
    # Handle any numerical issues
    heur = np.nan_to_num(heur, nan=1e-9, posinf=1e-9, neginf=1e-9)
    
    # Ensure positive values
    heur[heur <= 0] = 1e-9
    
    return heur
