import numpy as np

def heuristics(prizes, distance_matrix, max_len):
    n = len(prizes)
    eps = 1e-9
    
    # 1. Compute Budget-Aware Base Heuristic
    # cost_ij = d_ij + d_j0
    # eta_ij = (prize_j / cost_ij) * exp(-lambda * cost_ij / max_len)
    # We vectorize this.
    # distance_matrix is (n, n)
    # d_j0 is distance_matrix[:, 0] -> shape (n,)
    # We need to add d_j0 to each row i.
    # d_j0[i] is the distance from node i to 0.
    # So cost[i, j] = distance_matrix[i, j] + distance_matrix[j, 0]
    
    d_j0 = distance_matrix[:, 0] # shape (n,)
    # cost[i, j] = distance_matrix[i, j] + d_j0[j]
    # d_j0[j] needs to be broadcast to (1, n)
    cost = distance_matrix + d_j0[None, :]
    
    # Avoid division by zero
    cost_safe = cost + eps
    
    # Prize of destination j
    # prizes is shape (n,), we need prizes[j] for each i, j
    # So we broadcast prizes to (1, n)
    p_j = prizes[None, :]
    
    # Base heuristic
    lambda_val = 1.0
    # eta = (p_j / cost_safe) * exp(-lambda_val * cost / max_len)
    base_eta = (p_j / cost_safe) * np.exp(-lambda_val * cost / max_len)
    
    # 2. Greedy Tour Construction (Budget-Aware)
    current = 0
    current_len = 0.0
    visited = np.zeros(n, dtype=bool)
    visited[0] = True
    path = [0]
    
    while True:
        best_j = -1
        best_score = -1.0
        
        for j in range(n):
            if visited[j]:
                continue
            # Check feasibility
            if current_len + distance_matrix[current, j] + distance_matrix[j, 0] <= max_len:
                # Calculate score consistent with heuristic
                c = distance_matrix[current, j] + distance_matrix[j, 0]
                score = (prizes[j] / (c + eps)) * np.exp(-lambda_val * c / max_len)
                if score > best_score:
                    best_score = score
                    best_j = j
                    
        if best_j == -1:
            break
            
        # Move to best_j
        path.append(best_j)
        visited[best_j] = True
        current_len += distance_matrix[current, best_j]
        current = best_j
        
    # 3. Enhance Heuristic with Greedy Path
    # Boost edges in the greedy path
    boost = 100.0
    for k in range(len(path) - 1):
        u = path[k]
        v = path[k+1]
        base_eta[u, v] *= boost
        
    # 4. Sanitize
    base_eta[~np.isfinite(base_eta)] = 1e-9
    base_eta[base_eta <= 0] = 1e-9
    
    return base_eta
