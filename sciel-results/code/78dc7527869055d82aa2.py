import numpy as np

def heuristics(prizes, distance_matrix, max_len):
    n = len(prizes)
    # 1. Base Heuristic
    # Avoid division by zero
    dist_safe = distance_matrix + 1e-9
    # Heuristic: Prize of destination / Distance from source
    # This encourages moving to high prize nodes that are close
    base_eta = np.zeros((n, n))
    for i in range(n):
        base_eta[i, :] = prizes / dist_safe[i, :]
        
    # 2. Greedy Tour Construction
    current = 0
    current_len = 0.0
    visited = np.zeros(n, dtype=bool)
    visited[0] = True
    path = [0]
    
    while True:
        best_j = -1
        best_score = -1.0
        
        # Find candidates
        # Condition: not visited AND current_len + d[current, j] + d[j, 0] <= max_len
        for j in range(n):
            if visited[j]:
                continue
            if current_len + distance_matrix[current, j] + distance_matrix[j, 0] <= max_len:
                # Score: Prize / Distance
                score = prizes[j] / (distance_matrix[current, j] + 1e-9)
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
    # Non-positive or non-finite values become 1e-9
    base_eta[~np.isfinite(base_eta)] = 1e-9
    base_eta[base_eta <= 0] = 1e-9
    
    return base_eta
