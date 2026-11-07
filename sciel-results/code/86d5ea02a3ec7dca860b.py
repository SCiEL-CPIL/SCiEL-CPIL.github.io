import numpy as np

def heuristics(distance_matrix, coordinates, demands, capacity):
    n = distance_matrix.shape[0]
    # distance_matrix is n x n
    # demands is length n, capacity is scalar
    
    # Initialize heuristic matrix
    H = np.zeros((n, n))
    
    # Precompute distances from depot (node 0) to all others
    d_depot = distance_matrix[0, :]
    
    # Compute savings for all pairs of customers (i, j) where i, j > 0
    # Savings S_ij = d(i, 0) + d(0, j) - d(i, j)
    # We only care about i != j
    savings = np.zeros((n, n))
    for i in range(1, n):
        for j in range(1, n):
            if i != j:
                savings[i, j] = d_depot[i] + d_depot[j] - distance_matrix[i, j]
    
    # Normalize savings to [0, 1] or similar
    # Only consider positive savings
    max_savings = np.max(savings)
    if max_savings > 0:
        normalized_savings = savings / max_savings
    else:
        normalized_savings = np.zeros_like(savings)
    
    # Fill heuristic matrix
    # 1. Depot to customer (i=0, j>0)
    for j in range(1, n):
        # Prefer closer customers to start? Or those with high potential savings?
        # Let's stick to inverse distance for start
        H[0, j] = 1.0 / (distance_matrix[0, j] + 1e-9)
        
    # 2. Customer to customer (i>0, j>0, i!=j)
    for i in range(1, n):
        for j in range(1, n):
            if i != j:
                # Combine savings and distance
                # High savings and short distance -> High attractiveness
                # If savings is 0, attractiveness is low
                # We use (1 + normalized_savings) to ensure positive
                # And divide by distance
                d_ij = distance_matrix[i, j]
                H[i, j] = (1.0 + normalized_savings[i, j]) / (d_ij + 1e-9)
                
    # 3. Handle any non-positive values
    H[H <= 0] = 1e-9
    
    # Ensure shape is correct
    return H
