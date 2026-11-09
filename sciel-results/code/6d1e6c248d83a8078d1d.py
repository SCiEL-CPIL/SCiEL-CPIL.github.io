"""Direct approach: explicit tensor decomposition for 4x5 * 5x5 matrix multiplication.

Uses the schoolbook method with rank 100 = 4*5*5.
Each rank-1 term corresponds to one (i,j,k) triple.
"""

import time

DIMENSIONS = [4, 5, 5]
TIME_LIMIT = 110.0  # Safety margin before 120s limit


def construct():
    """Return the schoolbook decomposition of rank 100 for 4x5 * 5x5 matrix multiplication."""
    start_time = time.time()
    
    n, m, p = DIMENSIONS
    rank = n * m * p  # 100
    
    U = [[0] * rank for _ in range(n * m)]  # (20, 100)
    V = [[0] * rank for _ in range(m * p)]  # (25, 100)
    W = [[0] * rank for _ in range(p * n)]  # (20, 100)
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    # Time guard
    if time.time() - start_time > TIME_LIMIT:
        pass  # Should not happen for direct construction
    
    return {"U": U, "V": V, "W": W}
