import numpy as np
import time
import random

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    target_rank = 17
    # Try to find a decomposition of rank 17
    # If not found, fall back to rank 18
    
    def verify(U, V, W):
        # U: (n*m, R), V: (m*p, R), W: (p*n, R)
        # Check if sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i,j,k
        R = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(R):
                        s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if s != 1:
                        return False
        return True

    # Initialize with rank 18
    rank_18 = n * m * p
    U18 = np.zeros((n*m, rank_18), dtype=int)
    V18 = np.zeros((m*p, rank_18), dtype=int)
    W18 = np.zeros((p*n, rank_18), dtype=int)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U18[i*m+j, col] = 1
                V18[j*p+k, col] = 1
                W18[k*n+i, col] = 1
                col += 1
    
    # Try to find rank 17
    # Use a simple randomized search
    start_time = time.time()
    timeout = 10  # seconds
    
    while time.time() - start_time < timeout:
        # Randomly perturb the rank 18 solution to try to merge two terms
        # Or just random search with small integers
        U = np.random.randint(-2, 3, size=(n*m, target_rank))
        V = np.random.randint(-2, 3, size=(m*p, target_rank))
        W = np.random.randint(-2, 3, size=(p*n, target_rank))
        
        if verify(U, V, W):
            # Convert to list of lists
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }
            
    # Fallback to rank 18
    return {
        "U": U18.tolist(),
        "V": V18.tolist(),
        "W": W18.tolist()
    }
