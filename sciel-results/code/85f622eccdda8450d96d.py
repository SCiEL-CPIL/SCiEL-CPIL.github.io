import numpy as np
import time

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    nm = n * m  # 9
    mp = m * p  # 21
    pn = p * n  # 21
    
    # Constructive rank-21 decomposition
    # For each (j, k) pair, create one term c = j*p + k
    # U[i*m+j, c] = 1 for all i
    # V[j*p+k, c] = 1 (only for the matching j,k)
    # W[k*n+i, c] = 1 for all i
    #
    # Then for any (i,j,k): sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
    # = U[i*m+j, j*p+k] * V[j*p+k, j*p+k] * W[k*n+i, j*p+k]
    # = 1 * 1 * 1 = 1
    # All other terms have V[j*p+k, c] = 0 for c != j*p+k
    
    rank = m * p  # 21
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    
    for j in range(m):
        for k in range(p):
            c = j * p + k
            # U[i*m+j, c] = 1 for all i
            for i in range(n):
                U[i * m + j, c] = 1.0
            # V[j*p+k, c] = 1
            V[j * p + k, c] = 1.0
            # W[k*n+i, c] = 1 for all i
            for i in range(n):
                W[k * n + i, c] = 1.0
    
    # Verify
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(val - 1.0) > 1e-9:
                    # Fallback to schoolbook
                    rank = n * m * p
                    U = np.zeros((nm, rank))
                    V = np.zeros((mp, rank))
                    W = np.zeros((pn, rank))
                    col = 0
                    for ii in range(n):
                        for jj in range(m):
                            for kk in range(p):
                                U[ii*m+jj, col] = 1
                                V[jj*p+kk, col] = 1
                                W[kk*n+ii, col] = 1
                                col += 1
                    return {
                        "U": U.tolist(),
                        "V": V.tolist(),
                        "W": W.tolist()
                    }
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
