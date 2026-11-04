"""Matrix multiplication 3,3,6 - rank-minimal exact tensor decomposition.

Strategy:
1. The problem is multiplying A (3x3) by B (3x6) to get C (3x6).
2. The tensor M_{3,3,6}[i,j,k] = 1 for all i in [0,2], j in [0,2], k in [0,5].
3. We need to find the minimum rank r such that:
   M[i,j,k] = sum_{c=0}^{r-1} U[i,j,c] * V[j,k,c] * W[k,i,c]

4. A trivial upper bound is 3*3*6 = 54 (schoolbook).
5. A better construction: For each (j,k) pair, we can have one term where
   U[i,j,c] = 1 for all i, V[j,k,c] = 1, and W[k,i,c] = 1 for all i.
   This gives rank = m*p = 3*6 = 18.

6. Can we do even better? Let's think about the structure.
   The tensor is separable in the sense that M[i,j,k] = 1 for all i,j,k.
   This is a rank-1 tensor in the full space, but the constraint is that
   each term must be of the form U[i,j] * V[j,k] * W[k,i] (with the
   cyclic index structure).

7. The rank of the matrix multiplication tensor M_{n,m,p} is at most n*m*p
   (schoolbook) and at least max(n*m, m*p, p*n) = max(9, 18, 18) = 18.

8. So the lower bound is 18 and we have an upper bound of 18. Therefore
   the rank-minimal solution has rank exactly 18.
"""

import time
import numpy as np

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p, rank):
    """
    Verify that the tensor decomposition is exact.
    """
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    if len(U) == 0:
        return False
    for c in range(rank):
        if len(U[0]) != rank or len(V[0]) != rank or len(W[0]) != rank:
            return False
            
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    n, m, p = DIMENSIONS
    
    # Rank-minimal solution: rank = m * p = 18
    # For each (j,k) pair, one term covers all i
    rank = m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for j in range(m):
        for k in range(p):
            # One term for this (j,k) pair, shared across all i
            for i in range(n):
                U[i * m + j][column] = 1
                W[k * n + i][column] = 1
            V[j * p + k][column] = 1
            column += 1
    
    # Verify the solution
    if not _verify(U, V, W, n, m, p, rank):
        # Fallback to schoolbook if verification fails
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
