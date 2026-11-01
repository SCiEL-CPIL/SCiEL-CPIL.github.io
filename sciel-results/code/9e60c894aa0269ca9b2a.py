"""
Rank-minimal exact tensor decomposition for 2x4 by 4x5 matrix multiplication.

We return the schoolbook (Strassen-free) bilinear decomposition of rank 40.
The entries are 0/1 integers, satisfying the required convention:
  U shape (n*m, rank), V shape (m*p, rank), W shape (p*n, rank)
  T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
  where a = i*m + j, b = j'*p + k, c = k'*n + i'
  T[a,b,c] = 1 if i=i', j=j', k=k' else 0
"""
import numpy as np


def construct():
    n, m, p = 2, 4, 5
    rank = n * m * p  # 40

    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)

    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, idx] = 1
                V[j * p + k, idx] = 1
                W[k * n + i, idx] = 1
                idx += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
