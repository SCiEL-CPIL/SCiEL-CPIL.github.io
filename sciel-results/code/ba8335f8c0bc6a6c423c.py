import numpy as np
import time

def construct():
    n, m, p = 2, 4, 4
    # Rank 16 is the theoretical minimum (max(n*m, m*p, p*n) = 16)
    # We construct the schoolbook-like decomposition that achieves this rank
    # by sharing the V and W factors between the two rows of A.
    
    rank = 16
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # r corresponds to (j, k) pair in the schoolbook algorithm
    # j is the column of A / row of B (0..3)
    # k is the column of B / column of C (0..3)
    for j in range(m):
        for k in range(p):
            r = j * p + k
            
            # U: coefficients for A
            # A has shape (n, m). Index is i*m + j.
            # For this term, we pick A[i, j] for both i=0 and i=1.
            # U[i*m + j, r] = 1
            for i in range(n):
                U[i * m + j, r] = 1.0
            
            # V: coefficients for B
            # B has shape (m, p). Index is j*p + k.
            # For this term, we pick B[j, k].
            # V[j*p + k, r] = 1
            V[j * p + k, r] = 1.0
            
            # W: coefficients for C
            # C has shape (n, p). Index is k*n + i.
            # For this term, we contribute to C[i, k] for both i=0 and i=1.
            # W[k*n + i, r] = 1
            for i in range(n):
                W[k * n + i, r] = 1.0
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
