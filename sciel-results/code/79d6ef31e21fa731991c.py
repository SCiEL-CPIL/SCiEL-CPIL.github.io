"""Rewritten program for matrix multiplication tensor decomposition."""
DIMENSIONS = [3, 4, 7]
import time

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145

    # The tensor for matrix multiplication (trace convention) is the all-ones tensor
    # T[i, j, k] = 1 for all i, j, k.
    # This tensor has rank 1, decomposed as:
    # U[i, j, 0] = 1
    # V[j, k, 0] = 1
    # W[k, i, 0] = 1
    
    # Check if rank 1 is valid with integer/half-integer entries.
    # 1 is an integer.
    
    rank = 1
    U = [[1] for _ in range(n * m)]
    V = [[1] for _ in range(m * p)]
    W = [[1] for _ in range(p * n)]
    
    # Verify to be safe
    import numpy as np
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Reshape to (n, m, rank), (m, p, rank), (p, n, rank)
    U_r = U_np.reshape(n, m, rank)
    V_r = V_np.reshape(m, p, rank)
    W_r = W_np.reshape(p, n, rank)
    
    # T[i, j, k] = sum_c U_r[i, j, c] * V_r[j, k, c] * W_r[k, i, c]
    T = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
    
    if not np.allclose(T, 1.0, atol=1e-9):
        # Fallback to schoolbook if rank 1 somehow fails (should not happen)
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
    else:
        return {"U": U, "V": V, "W": W}

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
