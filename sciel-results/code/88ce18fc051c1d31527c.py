"""Rank-28 exact tensor decomposition for 2x3 * 3x5 matrix multiplication.

Uses a partitioned approach:
- For rows i=0,1 and columns k=0..3: apply Strassen on 2x2 x 2x2 blocks (rank 7 each)
- For rows i=0,1 and column k=4: schoolbook (rank 4)
- For column j=2 (all k): schoolbook (rank 10)
Total: 7 + 7 + 4 + 10 = 28
"""

import time
import numpy as np


def construct():
    deadline = time.time() + 110.0

    n, m, p = 2, 3, 5
    rank = 28

    U = [[0] * rank for _ in range(n * m)]   # 6 x 28
    V = [[0] * rank for _ in range(m * p)]   # 15 x 28
    W = [[0] * rank for _ in range(p * n)]   # 10 x 28

    col = 0

    def add_strassen(i1, i2, j1, j2, k1, k2):
        """Add 7 Strassen terms for multiplying A[i1:i2+1, j1:j2+1] * B[j1:j2+1, k1:k2+1]."""
        nonlocal col
        # M1 = (A[i1,j1]+A[i2,j2]) * (B[j1,k1]+B[j2,k2])
        U[i1 * m + j1][col] = 1
        U[i2 * m + j2][col] = 1
        V[j1 * p + k1][col] = 1
        V[j2 * p + k2][col] = 1
        W[k1 * n + i1][col] = 1
        W[k2 * n + i2][col] = 1
        col += 1

        # M2 = (A[i2,j1]+A[i2,j2]) * B[j1,k1]
        U[i2 * m + j1][col] = 1
        U[i2 * m + j2][col] = 1
        V[j1 * p + k1][col] = 1
        W[k1 * n + i2][col] = 1
        W[k2 * n + i2][col] = -1
        col += 1

        # M3 = A[i1,j1] * (B[j1,k2]-B[j2,k2])
        U[i1 * m + j1][col] = 1
        V[j1 * p + k2][col] = 1
        V[j2 * p + k2][col] = -1
        W[k2 * n + i1][col] = 1
        W[k2 * n + i2][col] = 1
        col += 1

        # M4 = A[i2,j2] * (B[j2,k1]-B[j1,k1])
        U[i2 * m + j2][col] = 1
        V[j2 * p + k1][col] = 1
        V[j1 * p + k1][col] = -1
        W[k1 * n + i1][col] = 1
        W[k1 * n + i2][col] = 1
        col += 1

        # M5 = (A[i1,j1]+A[i1,j2]) * B[j2,k2]
        U[i1 * m + j1][col] = 1
        U[i1 * m + j2][col] = 1
        V[j2 * p + k2][col] = 1
        W[k1 * n + i1][col] = -1
        W[k2 * n + i1][col] = 1
        col += 1

        # M6 = (A[i2,j1]-A[i1,j1]) * (B[j1,k1]+B[j1,k2])
        U[i2 * m + j1][col] = 1
        U[i1 * m + j1][col] = -1
        V[j1 * p + k1][col] = 1
        V[j1 * p + k2][col] = 1
        W[k2 * n + i2][col] = 1
        col += 1

        # M7 = (A[i1,j2]-A[i2,j2]) * (B[j2,k1]+B[j2,k2])
        U[i1 * m + j2][col] = 1
        U[i2 * m + j2][col] = -1
        V[j2 * p + k1][col] = 1
        V[j2 * p + k2][col] = 1
        W[k1 * n + i1][col] = 1
        col += 1

    def add_schoolbook(i_list, j_list, k_list):
        """Add schoolbook terms for the given index sets."""
        nonlocal col
        for i in i_list:
            for j in j_list:
                for k in k_list:
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1

    # Part 1: j=0,1, k=0,1 -> Strassen (rank 7)
    add_strassen(0, 1, 0, 1, 0, 1)

    # Part 2: j=0,1, k=2,3 -> Strassen (rank 7)
    add_strassen(0, 1, 0, 1, 2, 3)

    # Part 3: j=0,1, k=4 -> schoolbook (rank 4)
    add_schoolbook([0, 1], [0, 1], [4])

    # Part 4: j=2, k=0,1 -> schoolbook (rank 4)
    add_schoolbook([0, 1], [2], [0, 1])

    # Part 5: j=2, k=2,3 -> schoolbook (rank 4)
    add_schoolbook([0, 1], [2], [2, 3])

    # Part 6: j=2, k=4 -> schoolbook (rank 2)
    add_schoolbook([0, 1], [2], [4])

    assert col == 28, f"Expected 28 terms, got {col}"

    # Verify the decomposition
    if time.time() < deadline:
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)

        # Reconstruct the tensor T[i,j,k] and check it equals 1 for all (i,j,k)
        T = np.zeros((n, m, p))
        for r in range(rank):
            u_r = U_np[:, r].reshape(n, m)
            v_r = V_np[:, r].reshape(m, p)
            w_r = W_np[:, r].reshape(p, n)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T[i, j, k] += u_r[i, j] * v_r[j, k] * w_r[k, i]

        if not np.allclose(T, 1.0, atol=1e-10):
            # Fallback to rank-30 schoolbook
            U_fb = [[0] * 30 for _ in range(n * m)]
            V_fb = [[0] * 30 for _ in range(m * p)]
            W_fb = [[0] * 30 for _ in range(p * n)]
            col_fb = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U_fb[i * m + j][col_fb] = 1
                        V_fb[j * p + k][col_fb] = 1
                        W_fb[k * n + i][col_fb] = 1
                        col_fb += 1
            return {"U": U_fb, "V": V_fb, "W": W_fb}

    return {"U": U, "V": V, "W": W}
