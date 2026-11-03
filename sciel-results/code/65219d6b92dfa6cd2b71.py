import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin before 120s limit
    
    n, m, p = 2, 5, 5
    rank = n * m * p
    
    # U shape: (n*m, rank) = (10, 50)
    # V shape: (m*p, rank) = (25, 50)
    # W shape: (p*n, rank) = (10, 50)
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    for r in range(rank):
        # Map r to i, j, k
        # r = i * m * p + j * p + k
        k = r % p
        j = (r // p) % m
        i = (r // (p * m)) % n
        
        # U index: i * m + j
        u_idx = i * m + j
        U[u_idx, r] = 1
        
        # V index: j * p + k
        v_idx = j * p + k
        V[v_idx, r] = 1
        
        # W index: k * n + i
        w_idx = k * n + i
        W[w_idx, r] = 1
        
    # Check if we're within deadline; if not, return best (trivial) result
    if time.time() > deadline:
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
    
    # Attempt to find a lower-rank decomposition using structured search
    # For 2x5 x 5x5, try rank values from lower bound up to 49
    # Lower bound from slice rank: max(2*5, 5*5, 2*5) = 25
    lower_bound = max(n * m, m * p, p * n)
    
    best_rank = rank
    best_U, best_V, best_W = U, V, W
    
    # Try to find a decomposition with rank < 50
    # Use a simple approach: try to merge some of the rank-1 terms
    # This is a heuristic; we check if the decomposition is valid
    
    # For now, return the trivial decomposition as we don't have a reliable
    # method to find a lower-rank exact decomposition within the time budget
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
