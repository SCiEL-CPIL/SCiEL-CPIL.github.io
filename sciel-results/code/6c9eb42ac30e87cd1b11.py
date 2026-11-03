import numpy as np

def construct():
    n, m, p = 3, 4, 7
    
    # Split p into p1=3 and p2=4
    p1 = 3
    p2 = 4
    
    rank1 = n * m * p1
    rank2 = n * m * p2
    total_rank = rank1 + rank2
    
    U = np.zeros((n * m, total_rank), dtype=int)
    V = np.zeros((m * p, total_rank), dtype=int)
    W = np.zeros((p * n, total_rank), dtype=int)
    
    # Block 1: k in [0, p1)
    # r from 0 to rank1-1
    # r maps to i, j, k1
    for r in range(rank1):
        k1 = r % p1
        j = (r // p1) % m
        i = r // (p1 * m)
        
        # U index
        u_idx = i * m + j
        U[u_idx, r] = 1
        
        # V index
        # Global k = k1
        v_idx = j * p + k1
        V[v_idx, r] = 1
        
        # W index
        # Global k = k1
        w_idx = k1 * n + i
        W[w_idx, r] = 1
        
    # Block 2: k in [p1, p1+p2) i.e., [3, 7)
    # r from rank1 to total_rank-1
    # Local rank r2 from 0 to rank2-1
    for r2 in range(rank2):
        r = rank1 + r2
        k2 = r2 % p2
        j = (r2 // p2) % m
        i = r2 // (p2 * m)
        
        # U index
        # Same A, so same i,j
        u_idx = i * m + j
        U[u_idx, r] = 1
        
        # V index
        # Global k = p1 + k2
        v_idx = j * p + (p1 + k2)
        V[v_idx, r] = 1
        
        # W index
        # Global k = p1 + k2
        w_idx = (p1 + k2) * n + i
        W[w_idx, r] = 1

    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
