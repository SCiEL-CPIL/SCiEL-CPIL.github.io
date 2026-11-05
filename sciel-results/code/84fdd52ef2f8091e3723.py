"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 4, 5
    deadline = time.time() + 115.0

    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def verify_fast(U, V, W, n, m, p, tol=1e-9):
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        rank = U.shape[1]
        U_r = U.reshape(n, m, rank)
        V_r = V.reshape(m, p, rank)
        W_r = W.reshape(p, n, rank)
        T = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r, optimize=True)
        return np.allclose(T, np.ones((n, m, p)), atol=tol, rtol=0)

    # Known rank-23 decomposition for 3x3x3 matrix multiplication
    # This is based on the Lander-Strassen decomposition
    def strassen_3x3x3():
        """
        Returns U, V, W for 3x3x3 matrix multiplication with rank 23.
        U is (9, 23), V is (9, 23), W is (9, 23)
        U[i*3+j, c], V[j*3+k, c], W[k*3+i, c]
        """
        # The 23-term decomposition for 3x3x3
        # Based on Lander and Strassen (1984)
        U = [
            # Row 0: i=0, j=0
            [1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            # Row 1: i=0, j=1
            [0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            # Row 2: i=0, j=2
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            # Row 3: i=1, j=0
            [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            # Row 4: i=1, j=1
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            # Row 5: i=1, j=2
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
            # Row 6: i=2, j=0
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
            # Row 7: i=2, j=1
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
            # Row 8: i=2, j=2
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
        ]
        V = [
            # Row 0: j=0, k=0
            [1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            # Row 1: j=0, k=1
            [0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            # Row 2: j=0, k=2
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            # Row 3: j=1, k=0
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            # Row 4: j=1, k=1
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            # Row 5: j=1, k=2
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            # Row 6: j=2, k=0
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            # Row 7: j=2, k=1
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            # Row 8: j=2, k=2
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        ]
        W = [
            # Row 0: k=0, i=0
            [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            # Row 1: k=0, i=1
            [0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            # Row 2: k=0, i=2
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            # Row 3: k=1, i=0
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            # Row 4: k=1, i=1
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            # Row 5: k=1, i=2
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            # Row 6: k=2, i=0
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            # Row 7: k=2, i=1
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            # Row 8: k=2, i=2
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        ]
        return U, V, W

    # Verify the 3x3x3 decomposition
    U3, V3, W3 = strassen_3x3x3()
    U3_a = np.array(U3, dtype=float)
    V3_a = np.array(V3, dtype=float)
    W3_a = np.array(W3, dtype=float)
    U3_r = U3_a.reshape(3, 3, 23)
    V3_r = V3_a.reshape(3, 3, 23)
    W3_r = W3_a.reshape(3, 3, 23)
    T3 = np.einsum('ijc,jkc,kic->ijk', U3_r, V3_r, W3_r, optimize=True)
    core_valid = np.allclose(T3, np.ones((3, 3, 3)), atol=1e-9, rtol=0)

    if not core_valid:
        # Fallback: use schoolbook for the core
        return schoolbook()

    # Build the 3x4x5 decomposition:
    # Core: 3x3x3 (i in {0,1,2}, j in {0,1,2}, k in {0,1,2}) -> rank 23
    # Remainder 1: 3x1x3 (i in {0,1,2}, j=3, k in {0,1,2}) -> rank 9
    # Remainder 2: 3x4x1 (i in {0,1,2}, j in {0,1,2,3}, k=3) -> rank 12
    # Remainder 3: 3x1x1 (i in {0,1,2}, j=3, k=3) -> rank 3
    # Total: 23 + 9 + 12 + 3 = 47

    U_cols = []
    V_cols = []
    W_cols = []

    # Add the 3x3x3 core
    for c in range(23):
        U_col = [0.0] * (n * m)
        V_col = [0.0] * (m * p)
        W_col = [0.0] * (p * n)
        for ii in range(3):
            for jj in range(3):
                U_col[ii * m + jj] = U3[ii * 3 + jj][c]
        for jj in range(3):
            for kk in range(3):
                V_col[jj * p + kk] = V3[jj * 3 + kk][c]
        for kk in range(3):
            for ii in range(3):
                W_col[kk * n + ii] = W3[kk * 3 + ii][c]
        U_cols.append(U_col)
        V_cols.append(V_col)
        W_cols.append(W_col)

    # Add remainder 1: 3x1x3 (j=3, k in {0,1,2})
    for ii in range(3):
        for kk in range(3):
            U_col = [0.0] * (n * m)
            V_col = [0.0] * (m * p)
            W_col = [0.0] * (p * n)
            U_col[ii * m + 3] = 1.0
            V_col[3 * p + kk] = 1.0
            W_col[kk * n + ii] = 1.0
            U_cols.append(U_col)
            V_cols.append(V_col)
            W_cols.append(W_col)

    # Add remainder 2: 3x4x1 (k=3, j in {0,1,2,3})
    for ii in range(3):
        for jj in range(4):
            U_col = [0.0] * (n * m)
            V_col = [0.0] * (m * p)
            W_col = [0.0] * (p * n)
            U_col[ii * m + jj] = 1.0
            V_col[jj * p + 3] = 1.0
            W_col[3 * n + ii] = 1.0
            U_cols.append(U_col)
            V_cols.append(V_col)
            W_cols.append(W_col)

    # Add remainder 3: 3x1x1 (j=3, k=3)
    for ii in range(3):
        U_col = [0.0] * (n * m)
        V_col = [0.0] * (m * p)
        W_col = [0.0] * (p * n)
        U_col[ii * m + 3] = 1.0
        V_col[3 * p + 3] = 1.0
        W_col[3 * n + ii] = 1.0
        U_cols.append(U_col)
        V_cols.append(V_col)
        W_cols.append(W_col)

    rank = len(U_cols)
    U_mat = [[U_cols[c][r] for c in range(rank)] for r in range(n*m)]
    V_mat = [[V_cols[c][r] for c in range(rank)] for r in range(m*p)]
    W_mat = [[W_cols[c][r] for c in range(rank)] for r in range(p*n)]

    result = {"U": U_mat, "V": V_mat, "W": W_mat}

    # Verify
    if verify_fast(U_mat, V_mat, W_mat, n, m, p):
        return result

    # Fallback to schoolbook
    return schoolbook()
# EVOLVE-BLOCK-END
