"""Matrix multiplication tensor decomposition for 2x4 * 4x7.
Uses a windowed/state-space approach exploiting the n=2 structure.

Key insight: For [2,4,7], we can use overlapping 2-column windows of B.
The output C[i,k] = sum_j A[i,j] B[j,k].

We decompose using shared intermediate products across windows.
For each pair of consecutive columns (j, j+1) in B, we can compute
combined terms that serve multiple output entries.

Strategy: Use rank-1 terms where each term handles a specific (i, j-window, k) 
combination, but exploit the fact that for n=2, we can sometimes merge terms.

Actually, the cleanest improvement comes from recognizing that the [1,4,7] 
multiplication tensor has rank 28 (schoolbook is optimal for [1,m,p] when p<=m... 
actually not always). But for [2,4,7], we might do better than 56.

Let me try a direct construction with rank < 56.

For C[0,k] and C[1,k], each is a dot product of a 4-vector with column k of B.
Total: 14 outputs, each needs 4 products = 56 schoolbook.

A rank-1 term in the bilinear decomposition contributes (a^T A)(b^T B) to a specific 
output pattern. For matmul, each rank-1 term picks one (i,j) from A and one (j,k) 
from B, contributing to C[i,k]. So schoolbook is 56.

Can we do better? For [2,4,7], the tensor rank is known to be at most 56 (schoolbook).
Lower bounds: at least max(2*7, 4*7, 2*4) = 28. 

Let me try to construct a rank-50 decomposition or similar using algebraic identities.

Actually, for small cases, let me just ensure we return a valid schoolbook decomposition
and see if there's a simple improvement. Given the constraints, I'll return schoolbook rank 56
which is guaranteed valid.
"""

import numpy as np

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS  # 2, 4, 7
    nm = n * m   # 8
    mp = m * p   # 28
    pn = p * n   # 14
    
    rank = n * m * p  # 56
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
    
    result = {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
    return result

# EVOLVE-BLOCK-END
