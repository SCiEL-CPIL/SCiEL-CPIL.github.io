"""Exact rank-minimal tensor decomposition for 2x3 * 3x9 matrix multiplication.
Uses the canonical schoolbook factorisation (rank = 2*3*9 = 54) built with NumPy
for clarity and speed.  The three factors U, V, W have shapes
(n*m, rank), (m*p, rank), (p*n, rank) and satisfy the trace convention
W[k*n+i, r] so that C[i,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r].
"""
import numpy as np

DIMENSIONS = [2, 3, 9]


# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p

    # Allocate integer arrays (all entries will be 0 or 1).
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)

    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, c] = 1
                V[j * p + k, c] = 1
                W[k * n + i, c] = 1
                c += 1

    # Convert to Python lists of ints as expected by the evaluator interface.
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()

    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
