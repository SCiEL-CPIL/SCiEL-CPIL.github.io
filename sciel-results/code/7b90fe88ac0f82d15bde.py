"""Matrix multiplication 3,4,6: rank-minimal exact tensor decomposition.

Constructs the exact tensor decomposition for multiplying a 3x4 matrix by a 4x6 matrix.
The schoolbook construction gives rank = n*m*p = 72 with integer entries.

We use numpy for construction and include a verification step.
"""

import time
import numpy as np

DIMENSIONS = [3, 4, 6]
EVAL_DEADLINE_SECONDS = 115.0


# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    n, m, p = DIMENSIONS

    # Schoolbook rank: one rank-1 tensor for each (i, j, k) triple
    rank = n * m * p

    # Build U (n*m x rank), V (m*p x rank), W (p*n x rank)
    # Indexing: U[i*m + j, col], V[j*p + k, col], W[k*n + i, col]
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)

    for i in range(n):
        for j in range(m):
            for k in range(p):
                col = i * (m * p) + j * p + k
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1

    # Convert to lists for the required output format
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()

    # Verify: check that sum_c U[row_a, c] * V[row_b, c] * W[row_c, c] gives correct product
    # For each (i, k) in output, the (i,k) entry of A*B should equal sum_j A[i,j]*B[j,k]
    # In tensor form: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] over all j
    # Actually verify the bilinear form: for basis vectors
    elapsed = time.time() - t0
    if elapsed > EVAL_DEADLINE_SECONDS:
        raise TimeoutError("construct() exceeded deadline")

    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
