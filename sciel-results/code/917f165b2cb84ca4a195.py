import numpy as np
import time

def construct():
    n, m, p = 3, 3, 3
    # Standard 27-term decomposition
    # C_ij = sum_k A_ik B_kj
    # U index: (i, k) -> i*m + k
    # V index: (k, j) -> k*p + j
    # W index: (j, i) -> j*n + i  (based on W[k*n+i] convention where k corresponds to j and i to i)
    
    # Rank is 27
    rank = 27
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    r = 0
    for i in range(n):
        for k in range(m):
            for j in range(p):
                # U
                idx_u = i * m + k
                U[idx_u, r] = 1.0
                
                # V
                idx_v = k * p + j
                V[idx_v, r] = 1.0
                
                # W
                # Convention: W[k*n+i, r] where k is row of W (size p), i is col of W (size n)
                # We want to store C_ij. 
                # If W is p x n, and index is k*n + i, then W[k, i] = C_ij implies k=j, i=i.
                idx_w = j * n + i
                W[idx_w, r] = 1.0
                
                r += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
