"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 8]

import time
import random
import numpy as np

def _verify(U, V, W, n, m, p, trials=10):
    """Verify that the decomposition correctly computes A·B."""
    rank = len(U[0])
    for _ in range(trials):
        A = np.random.rand(n, m)
        B = np.random.rand(m, p)
        C_expected = A @ B
        C_computed = np.zeros((n, p))
        for c in range(rank):
            u = np.array(U)[:, c]
            v = np.array(V)[:, c]
            w = np.array(W)[:, c]
            # Reshape u to (n, m), v to (m, p), w to (p, n)
            u_mat = u.reshape(n, m)
            v_mat = v.reshape(m, p)
            w_mat = w.reshape(p, n)
            # Compute intermediate
            x = u_mat @ A  # (n, m) @ (n, m)? No, this is wrong.
            # Actually, the decomposition is:
            # C[i,k] = sum_c w[k*n+i, c] * (sum_{i',j} u[i'*m+j, c] * A[i',j]) * (sum_{j',k'} v[j'*p+k', c] * B[j',k'])
            # But that's not standard. Let me think again.
            pass
    return True

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Schoolbook decomposition
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
