import numpy as np


def heuristics(prizes, distance_matrix, max_len):
    n = len(prizes)
    heuristic = np.ones((n, n)) * 1e-9
    
    # For each node i, find all nodes k that are reachable from i within budget
    # i.e., d[i,k] + d[k,0] <= max_len
    reachable = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for k in range(n):
            if k != i and k != 0:
                if distance_matrix[i, k] + distance_matrix[k, 0] <= max_len:
                    reachable[i, k] = True
    
    # For each (i, j), compute heuristic based on regret
    # Regret = Max Prize of Alternatives
    # Attractiveness = Prize[j] / (Regret + epsilon)
    eps = 1e-9
    for i in range(n):
        for j in range(n):
            if j == i or j == 0:
                heuristic[i, j] = 1e-9
                continue
            # Check if j is reachable from i
            if not reachable[i, j]:
                heuristic[i, j] = 1e-9
                continue
            
            # Find max prize among other reachable nodes k (k != j, k != i, k != 0)
            max_alt_prize = 0.0
            for k in range(n):
                if k != i and k != j and k != 0 and reachable[i, k]:
                    if prizes[k] > max_alt_prize:
                        max_alt_prize = prizes[k]
            
            # Calculate attractiveness
            # If prize[j] is 0, attractiveness is 0 (becomes 1e-9 later)
            val = prizes[j] / (max_alt_prize + eps)
            
            if val > 0 and np.isfinite(val):
                heuristic[i, j] = val
            else:
                heuristic[i, j] = 1e-9
    
    # Ensure non-finite values become 1e-9
    heuristic[~np.isfinite(heuristic)] = 1e-9
    heuristic[heuristic <= 0] = 1e-9
    
    return heuristic
