DIMENSIONS = [2, 4, 8]

import time
_START_TIME = time.time()
_DEADLINE = 110  # seconds, safety margin for 120s limit

def construct():
    n, m, p = DIMENSIONS
    # Standard Kronecker product decomposition has rank n*m*p
    rank = n * m * p
    
    # Initialize matrices
    # U shape: (n*m, rank)
    # V shape: (m*p, rank)
    # W shape: (p*n, rank)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # The rank index r corresponds to a triple (i, j, k)
    # We iterate over all i in [0, n), j in [0, m), k in [0, p)
    # and assign a unique rank index r to each triple.
    
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: (i, j) -> i*m + j
                u_idx = i * m + j
                U[u_idx][r] = 1
                
                # V index: (j, k) -> j*p + k
                v_idx = j * p + k
                V[v_idx][r] = 1
                
                # W index: (k, i) -> k*n + i (Trace convention W[k*n+i])
                w_idx = k * n + i
                W[w_idx][r] = 1
                
                r += 1
                
                # Check time
                if time.time() - _START_TIME > _DEADLINE:
                    return {"U": U, "V": V, "W": W}

    return {"U": U, "V": V, "W": W}
