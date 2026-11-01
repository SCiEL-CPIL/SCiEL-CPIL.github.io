import numpy as np

DIMENSIONS = [2, 2, 14]

def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize U, V, W as zero matrices
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank)
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # We iterate over the rank index r
    # r corresponds to a specific (i, j, k) triple
    # r = (i * m + j) * p + k
    # Or simply r from 0 to rank-1, and decompose r into i, j, k
    
    for r in range(rank):
        # Decompose r into i, j, k
        # r = i * (m * p) + j * p + k
        k = r % p
        tmp = r // p
        j = tmp % m
        i = tmp // m
        
        # Set the corresponding entries to 1
        u_idx = i * m + j
        v_idx = j * p + k
        w_idx = k * n + i
        
        U[u_idx, r] = 1
        V[v_idx, r] = 1
        W[w_idx, r] = 1
        
    # Convert to lists as required by the interface
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
