import numpy as np
import time
import random

DIMENSIONS = [2, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 50
    
    # Try to find a rank 39 decomposition
    # Split into (2x5 by 5x2) and (2x5 by 5x3)
    # Rank 15 and Rank 24
    
    def find_decomposition(n, m, p, target_rank, time_limit=10):
        """
        Find a decomposition of rank target_rank for n x m by m x p.
        Returns U, V, W or None.
        """
        start_time = time.time()
        # Use random search with integer/half-integer values
        # For small ranks, we can try random integers in {-1, 0, 1}
        
        # We need T[i*m+j, j*p+k, k*n+i] = 1
        # sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] = 1
        
        # For R(2, 5, 2) = 15
        # For R(2, 5, 3) = 24
        
        for attempt in range(1000):
            if time.time() - start_time > time_limit:
                return None
            
            U = np.random.randint(-1, 2, (n*m, target_rank))
            V = np.random.randint(-1, 2, (m*p, target_rank))
            W = np.random.randint(-1, 2, (p*n, target_rank))
            
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        a = i * m + j
                        b = j * p + k
                        c = k * n + i
                        val = np.sum(U[a, :] * V[b, :] * W[c, :])
                        if val != 1:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            if valid:
                return U, V, W
        
        return None

    # Try to find R(2, 5, 2) = 15
    res_15 = find_decomposition(2, 5, 2, 15, time_limit=5)
    if res_15:
        U1, V1, W1 = res_15
        # Try to find R(2, 5, 3) = 24
        res_24 = find_decomposition(2, 5, 3, 24, time_limit=10)
        if res_24:
            U2, V2, W2 = res_24
            # Combine
            # B is 5x5. Split into B1 (5x2) and B2 (5x3).
            # V1 is for B1 (5x2). Indices j*p1 + k1. p1=2.
            # V2 is for B2 (5x3). Indices j*p2 + k2. p2=3.
            # We need to map these to the full 5x5 B.
            # B indices: j*5 + k.
            # B1 indices: j*2 + k1 (k1 in 0,1). Maps to j*5 + k1.
            # B2 indices: j*3 + k2 (k2 in 0,1,2). Maps to j*5 + (2+k2).
            
            U = np.hstack([U1, U2])
            W = np.hstack([W1, W2])
            
            V_full = np.zeros((25, 39))
            for j in range(5):
                for k1 in range(2):
                    V_full[j*5 + k1, :15] = V1[j*2 + k1, :]
                for k2 in range(3):
                    V_full[j*5 + 2 + k2, 15:] = V2[j*3 + k2, :]
            
            return {
                "U": U.tolist(),
                "V": V_full.tolist(),
                "W": W.tolist()
            }

    # Fallback to schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
