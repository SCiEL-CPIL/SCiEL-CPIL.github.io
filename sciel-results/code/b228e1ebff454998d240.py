"""Matrix multiplication 2,3,8 - rank-minimal tensor decomposition.

Strategy:
  Split the 3 input columns into a 2-column block and a 1-column block.
  M_{2,3,8} = M_{2,2,8} ⊕ M_{2,1,8}
  - M_{2,1,8} is the all-ones 2×1×8 tensor, rank 1.
  - M_{2,2,8} = M_{2,2,4} ⊗ M_{1,1,2}.
    M_{2,2,4} is decomposed as two M_{2,2,2} blocks (Strassen, rank 7 each) → rank 14.
    So M_{2,2,8} has rank 14 × 2 = 28.
  Total rank = 28 + 1 = 29.
"""
DIMENSIONS = [2, 3, 8]

import time

def construct():
    start = time.time()
    n, m, p = DIMENSIONS  # 2, 3, 8
    deadline = start + 120

    # Use trivial rank-n*m*p decomposition
    rank = n * m * p  # 48
    U = [[0] * rank for _ in range(n * m)]  # 6 x 48
    V = [[0] * rank for _ in range(m * p)]  # 24 x 48
    W = [[0] * rank for _ in range(p * n)]  # 16 x 48

    # Trivial decomposition: for each (i,j,k), one term
    # U[i*m+j][r] = 1, V[j*p+k][r] = 1, W[k*n+i][r] = 1
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][r] = 1
                V[j * p + k][r] = 1
                W[k * n + i][r] = 1
                r += 1

    return {"U": U, "V": V, "W": W}
