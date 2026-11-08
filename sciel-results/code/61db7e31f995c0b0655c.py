"""Tensor decomposition for 2x5 by 5x6 matrix multiplication."""
import time

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130  # 130s limit, 20s margin for 150s kill

    n, m, p = DIMENSIONS
    rank = 48
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0
    j_pairs = [(0, 1), (2, 3)]
    k_pairs = [(0, 1), (2, 3), (4, 5)]

    for j0, j1 in j_pairs:
        for k0, k1 in k_pairs:
            # Strassen 7 products
            # P1 = (A_0,j0 + A_1,j1)(B_j0,k0 + B_j1,k1)
            U[0*m + j0][col] += 1
            U[1*m + j1][col] += 1
            V[j0*p + k0][col] += 1
            V[j1*p + k1][col] += 1
            W[k0*n + 0][col] += 1
            W[k1*n + 1][col] += 1
            col += 1

            # P2 = (A_1,j0 + A_1,j1) B_j0,k0
            U[1*m + j0][col] += 1
            U[1*m + j1][col] += 1
            V[j0*p + k0][col] += 1
            W[k0*n + 1][col] += 1
            W[k1*n + 1][col] -= 1
            col += 1

            # P3 = A_0,j0 (B_j0,k1 - B_j1,k1)
            U[0*m + j0][col] += 1
            V[j0*p + k1][col] += 1
            V[j1*p + k1][col] -= 1
            W[k1*n + 0][col] += 1
            W[k1*n + 1][col] += 1
            col += 1

            # P4 = A_1,j1 (B_j1,k0 - B_j0,k0)
            U[1*m + j1][col] += 1
            V[j1*p + k0][col] += 1
            V[j0*p + k0][col] -= 1
            W[k0*n + 0][col] += 1
            W[k0*n + 1][col] += 1
            col += 1

            # P5 = (A_0,j0 + A_0,j1) B_j1,k1
            U[0*m + j0][col] += 1
            U[0*m + j1][col] += 1
            V[j1*p + k1][col] += 1
            W[k0*n + 0][col] -= 1
            W[k1*n + 0][col] += 1
            col += 1

            # P6 = (A_1,j0 - A_0,j0)(B_j0,k0 + B_j0,k1)
            U[1*m + j0][col] += 1
            U[0*m + j0][col] -= 1
            V[j0*p + k0][col] += 1
            V[j0*p + k1][col] += 1
            W[k1*n + 1][col] += 1
            col += 1

            # P7 = (A_0,j1 - A_1,j1)(B_j1,k0 + B_j1,k1)
            U[0*m + j1][col] += 1
            U[1*m + j1][col] -= 1
            V[j1*p + k0][col] += 1
            V[j1*p + k1][col] += 1
            W[k0*n + 0][col] += 1
            col += 1

    # Remaining j=4
    for k in range(p):
        # C_0,k = A_0,4 B_4,k
        U[0*m + 4][col] += 1
        V[4*p + k][col] += 1
        W[k*n + 0][col] += 1
        col += 1

        # C_1,k = A_1,4 B_4,k
        U[1*m + 4][col] += 1
        V[4*p + k][col] += 1
        W[k*n + 1][col] += 1
        col += 1

    # Verify correctness
    import numpy as np
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)

    tensor_decomp = np.zeros((n, m, p))
    tensor_true = np.zeros((n, m, p))

    for i in range(n):
        for j in range(m):
            for k in range(p):
                tensor_true[i, j, k] = 1.0
                for r in range(rank):
                    tensor_decomp[i, j, k] += U_arr[i*m + j, r] * V_arr[j*p + k, r] * W_arr[k*n + i, r]

    if not np.allclose(tensor_decomp, tensor_true, atol=1e-10):
        # Fallback: naive rank mn*p decomposition
        rank_naive = n * m * p
        U_naive = [[0] * rank_naive for _ in range(n * m)]
        V_naive = [[0] * rank_naive for _ in range(m * p)]
        W_naive = [[0] * rank_naive for _ in range(p * n)]
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_naive[i*m + j][idx] = 1
                    V_naive[j*p + k][idx] = 1
                    W_naive[k*n + i][idx] = 1
                    idx += 1
        return {"U": U_naive, "V": V_naive, "W": W_naive}

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
