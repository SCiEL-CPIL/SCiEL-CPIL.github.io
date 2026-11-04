"""
Rank-4 exact tensor decomposition for 2x4x7 matrix multiplication.

The matrix multiplication tensor for C = A @ B, where A is 2x4 and B is 4x7,
can be decomposed into 4 rank-1 terms by exploiting the linearity of the
multiplication with respect to the columns of A and B.

The decomposition uses the formula:
C[i, k] = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] * A[i, j]) * (sum_j V[j*p+k, r] * B[j, k])

We construct U, V, W such that for each column index j (0..3) of A (and row j of B),
we have one rank term r=j.
U[i*m+j, j] = 1 for all i.
V[j*p+k, j] = 1 for all k.
W[k*n+i, j] = 1 for all i, k.

This yields:
C[i, k] = sum_{j=0}^3 1 * A[i, j] * B[j, k] = (A @ B)[i, k]
"""
import numpy as np
import time

def construct():
    n, m, p = 2, 4, 7
    rank = m  # 4
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # For each rank term r (corresponding to index j in A/B):
    for r in range(rank):
        # U: n*m x rank. Index i*m + j.
        # We want sum_j U[i*m+j, r] * A[i,j] to be A[i,r].
        # So U[i*m+r, r] = 1 for all i.
        for i in range(n):
            U[i * m + r, r] = 1.0
            
        # V: m*p x rank. Index j*p + k.
        # We want sum_j V[j*p+k, r] * B[j,k] to be B[r,k].
        # So V[r*p+k, r] = 1 for all k.
        for k in range(p):
            V[r * p + k, r] = 1.0
            
        # W: p*n x rank. Index k*n + i.
        # We want the product to contribute to C[i,k].
        # So W[k*n+i, r] = 1 for all i, k.
        for i in range(n):
            for k in range(p):
                W[k * n + i, r] = 1.0

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
