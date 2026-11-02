"""Matrix multiplication tensor decomposition 3x5x7 - Rank 105."""
import numpy as np
import time

def construct():
    t0 = time.time()
    deadline = t0 + 110  # safety margin before 120s limit
    
    n, m, p = 3, 5, 7
    rank = n * m * p  # 105, schoolbook
    
    # Initialize matrices
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Fill the decomposition
    # The tensor T_{i,j,k} is decomposed as sum_r U_{i*m+j, r} V_{j*p+k, r} W_{k*n+i, r}
    # For schoolbook, each (i,j,k) is a separate rank-1 term.
    for r in range(rank):
        # Unpack r into i, j, k
        # r = i * (m * p) + j * p + k
        k = r % p
        tmp = r // p
        j = tmp % m
        i = tmp // m
        
        idx_u = i * m + j
        idx_v = j * p + k
        idx_w = k * n + i
        
        U[idx_u, r] = 1
        V[idx_v, r] = 1
        W[idx_w, r] = 1
        
    # Verify time guard - return immediately for this small case
    if time.time() > deadline:
        # Should not happen for rank 105, but guard just in case
        pass
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
