"""Program for M_{2,3,9} matrix multiplication tensor decomposition.

Uses the block structure: B = [B0 | B1 | B2], each Bt is 3x3.
C = [A*B0 | A*B1 | A*B2]

For each block, uses the schoolbook 2x3 * 3x3 multiplication (rank 18 per block).
Total rank: 3 * 18 = 54.

The W index follows the convention W[k*n + i, r] where k is the column index in C (0..8)
and i is the row index (0..1).
"""

import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    n, m, p = 2, 3, 9
    R = n * m * p  # 54 schoolbook rank
    
    U = [[0] * R for _ in range(n * m)]  # 6 x R
    V = [[0] * R for _ in range(m * p)]  # 27 x R
    W = [[0] * R for _ in range(p * n)]  # 18 x R
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U: A[i,j] -> index i*m + j
                U[i * m + j][idx] = 1
                # V: B[j,k] -> index j*p + k
                V[j * p + k][idx] = 1
                # W: C[i,k] -> index k*n + i (per convention W[k*n+i, r])
                W[k * n + i][idx] = 1
                idx += 1
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > deadline - start_time:
        # Should not happen for this simple construction
        pass
    
    return {"U": U, "V": V, "W": W}
