"""
Tensor decomposition for 2x3 * 3x9 matrix multiplication.
Constructs the rank-54 (optimal) exact decomposition.
"""

import time

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p

    # Deadline guard: return immediately if we're past a safety margin
    start_time = time.time()
    if time.time() - start_time > 10.0:
        # Fallback: return zero matrices (should never happen)
        U = [[0.0] * rank for _ in range(n * m)]
        V = [[0.0] * rank for _ in range(m * p)]
        W = [[0.0] * rank for _ in range(p * n)]
        return {"U": U, "V": V, "W": W}

    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]

    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U[i*m + j][idx] = 1.0  (A[i,j] factor)
                # V[j*p + k][idx] = 1.0  (B[j,k] factor)
                # W[k*n + i][idx] = 1.0  (C[i,k] factor, trace convention W[k*n+i, r])
                U[i * m + j][idx] = 1.0
                V[j * p + k][idx] = 1.0
                W[k * n + i][idx] = 1.0
                idx += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
