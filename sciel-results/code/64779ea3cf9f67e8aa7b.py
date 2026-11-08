import time
import numpy as np

def construct():
    n, m, p = 2, 3, 10
    
    # Naive decomposition
    # Rank = n * m * p
    rank = n * m * p
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    idx = 0
    for i in range(n):
        for k in range(m):
            for j in range(p):
                # U index: i*m + k
                U[i * m + k, idx] = 1
                # V index: k*p + j
                V[k * p + j, idx] = 1
                # W index: j*n + i (Trace convention W[k*n+i, r] -> here output is C[i,j]
                # Task says: "output factor follows the official trace convention W[k*n+i,r]"
                # Usually W maps to output matrix.
                # If output is C (n x p), index is i*p + j?
                # The task says W shape is (p*n, rank).
                # Convention W[k*n+i, r] suggests index is k*n + i.
                # If k corresponds to row of B (which is col of A? No, B is m x p).
                # Let's trace:
                # A is n x m. B is m x p. C is n x p.
                # Term: A[i,k] * B[k,j] * C[i,j].
                # U is n*m. Index i*m+k.
                # V is m*p. Index k*p+j.
                # W is p*n. Index ?
                # If W index is j*n + i, then W[j*n+i, r] = 1.
                # This matches the shape (p*n).
                W[j * n + i, idx] = 1
                idx += 1
                
    return {'U': U, 'V': V, 'W': W}
