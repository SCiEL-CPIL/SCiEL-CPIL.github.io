import numpy as np

def heuristics(prizes, distance_matrix, max_len):
    n = len(prizes)
    d = distance_matrix
    # Incremental cost of moving from i to j and returning to 0
    # Note: d[:, 0] is the return cost for each node j (column 0)
    # d[i, j] is the cost from i to j
    # total_incremental_cost[i, j] = d[i, j] + d[j, 0]
    return_cost = d[:, 0]
    incremental_cost = d + return_cost[None, :]
    
    # Gaussian heuristic
    # sigma scales with max_len to adapt to the budget
    sigma = max_len / 3.0
    # Avoid division by zero or issues if sigma is 0
    if sigma < 1e-6:
        sigma = 1e-6
        
    # Calculate exponential decay
    # exponent = - (cost^2) / (2 * sigma^2)
    exponent = - (incremental_cost ** 2) / (2.0 * sigma ** 2)
    decay = np.exp(exponent)
    
    # Combine with prize
    heuristic = prizes[None, :] * decay
    
    # Ensure non-finite and non-positive become 1e-9
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    return heuristic
