"""Matrix multiplication tensor decomposition for 2x4 * 4x7.

Exploits the n=2 structure by splitting A into top and bottom rows.
C_top = A_top @ B, C_bot = A_bot @ B.

We use the fact that:
C_bot = C_top + (A_bot - A_top) @ B

But more directly, we can use a structured decomposition that shares
intermediate products between the two rows.

For the schoolbook, rank = 2*4*7 = 56.

A better approach: for each column k of B, we need to compute:
C[0,k] = sum_j A[0,j] B[j,k]
C[1,k] = sum_j A[1,j] B[j,k]

We can combine the two outputs for each (j,k) pair:
Let S_j = A[0,j] + A[1,j], D_j = A[1,j] - A[0,j]
Then:
C[0,k] = sum_j (S_j - D_j)/2 * B[j,k]
C[1,k] = sum_j (S_j + D_j)/2 * B[j,k]

This doesn't directly help reduce rank since we still have 4*7 = 28 products per row.

Let me try a different approach: use the tensor decomposition framework directly.
The matmul tensor for [n,m,p] has rank at most n*m*p (schoolbook).
For [2,4,7], can we do better?

Known: For [2,m,p], there's a construction with rank <= m*p + m*p - m = 2*m*p - m
when we can share some products. For m=4, p=7: 2*4*7 - 4 = 52.

Actually, the standard improvement for [2,m,p] is:
- Compute A_top @ B (rank m*p)
- Compute (A_bot - A_top) @ B (rank m*p)
- C_bot = C_top + (A_bot - A_top) @ B (free, just addition)
Total rank: 2*m*p = 56 (no improvement)

But if we can share products between the two computations... 

For [2,4,7], let me try rank 52 by combining 4 of the products.
For each j, we have A[0,j]*B[j,k] and A[1,j]*B[j,k].
We can write:
(A[0,j] + A[1,j]) * B[j,k] and (A[1,j] - A[0,j]) * B[j,k]
Then A[0,j]*B[j,k] = ((A[0,j]+A[1,j]) - (A[1,j]-A[0,j]))/2 * B[j,k]
A[1,j]*B[j,k] = ((A[0,j]+A[1,j]) + (A[1,j]-A[0,j]))/2 * B[j,k]

This still requires 2 products per (j,k) pair, so no savings.

Let me just construct the schoolbook decomposition and verify it works.
"""

import numpy as np

def construct():
    n, m, p = 2, 4, 7
    nm = n * m   # 8
    mp = m * p   # 28
    pn = p * n   # 14
    
    rank = n * m * p  # 56 for schoolbook
    U = np.zeros((nm, rank), dtype=float)
    V = np.zeros((mp, rank), dtype=float)
    W = np.zeros((pn, rank), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
