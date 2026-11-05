"""Matrix multiplication 3x4x5 tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 4, 5
    deadline = time.time() + 115.0  # safety margin before 150s kill

    def strassen_2x2x2():
        """Return U, V, W for 2x2x2 Strassen decomposition."""
        U = [
            [1, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 0],
            [0, 0, 1, 0, 0, 1, 0],
            [1, 1, 0, 0, 0, 0, 1],
        ]
        V = [
            [1, 0, 0, 0, 0, 1, 1],
            [0, 1, -1, 1, 0, 0, 1],
            [1, 0, 1, 0, 0, 1, 0],
            [1, 1, 0, -1, 1, 0, 0],
        ]
        W = [
            [1, 0, 1, 0, 0, 0, 0],
            [0, 1, 0, 0, 1, 0, 1],
            [0, 0, 0, 1, 0, 1, 0],
            [1, 0, 0, 1, -1, 0, 0],
        ]
        return U, V, W

    def verify_fast(U, V, W, n, m, p, tol=1e-9):
        """Faster verification using einsum."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        rank = U.shape[1]
        U_r = U.reshape(n, m, rank)
        V_r = V.reshape(m, p, rank)
        W_r = W.reshape(p, n, rank)
        T = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r, optimize=True)
        return np.allclose(T, np.ones((n, m, p)), atol=tol, rtol=0)

    U_cols = []
    V_cols = []
    W_cols = []

    def add_2x2x5_block(i_offset, j_offset, k_offset):
        """Add a 2x2x5 block decomposition using Strassen for k-pairs and schoolbook for last k."""
        U_s, V_s, W_s = strassen_2x2x2()
        # k in {0,1} relative to k_offset: Strassen
        for c in range(7):
            U_col = [0.0] * (n * m)
            V_col = [0.0] * (m * p)
            W_col = [0.0] * (p * n)
            for ii in range(2):
                for jj in range(2):
                    U_col[(i_offset + ii) * m + (j_offset + jj)] = U_s[ii * 2 + jj][c]
            for jj in range(2):
                for kk in range(2):
                    V_col[(j_offset + jj) * p + (k_offset + kk)] = V_s[jj * 2 + kk][c]
            for kk in range(2):
                for ii in range(2):
                    W_col[(k_offset + kk) * n + (i_offset + ii)] = W_s[kk * 2 + ii][c]
            U_cols.append(U_col)
            V_cols.append(V_col)
            W_cols.append(W_col)

        # k in {2,3} relative to k_offset: Strassen
        for c in range(7):
            U_col = [0.0] * (n * m)
            V_col = [0.0] * (m * p)
            W_col = [0.0] * (p * n)
            for ii in range(2):
                for jj in range(2):
                    U_col[(i_offset + ii) * m + (j_offset + jj)] = U_s[ii * 2 + jj][c]
            for jj in range(2):
                for kk in range(2):
                    V_col[(j_offset + jj) * p + (k_offset + 2 + kk)] = V_s[jj * 2 + kk][c]
            for kk in range(2):
                for ii in range(2):
                    W_col[(k_offset + 2 + kk) * n + (i_offset + ii)] = W_s[kk * 2 + ii][c]
            U_cols.append(U_col)
            V_cols.append(V_col)
            W_cols.append(W_col)

        # k in {4} relative to k_offset: rank-2 decomposition of 2x2x1
        # C_{ii,jj} = A_{ii,jj} * B_{jj,4} for ii in {0,1}, jj in {0,1}
        # Term 1: A_{0,jj} * B_{jj,4} * delta_{ii,0}
        # Term 2: A_{1,jj} * B_{jj,4} * delta_{ii,1}
        for ii in range(2):
            U_col = [0.0] * (n * m)
            V_col = [0.0] * (m * p)
            W_col = [0.0] * (p * n)
            for jj in range(2):
                U_col[(i_offset + ii) * m + (j_offset + jj)] = 1.0
                V_col[(j_offset + jj) * p + (k_offset + 4)] = 1.0
            W_col[(k_offset + 4) * n + (i_offset + ii)] = 1.0
            U_cols.append(U_col)
            V_cols.append(V_col)
            W_cols.append(W_col)

    def add_1x2x5_block(i_idx, j_offset, k_offset):
        """Add a 1x2x5 block decomposition with rank 6."""
        # 1x2x2 for k in {0,1}: rank 2
        # C_{jj} = A_{jj} * B_{jj,0} + ... wait
        # For 1x2x2: A is 1x2 [a0,a1], B is 2x2, C is 1x2 [c0,c1]
        # c0 = a0*B00 + a1*B10
        # c1 = a0*B01 + a1*B11
        # Term 1: A_0 * [B00, B01] * [1, 1]
        # Term 2: A_1 * [B10, B11] * [1, 1]
        for jj in range(2):
            U_col = [0.0] * (n * m)
            V_col = [0.0] * (m * p)
            W_col = [0.0] * (p * n)
            U_col[i_idx * m + (j_offset + jj)] = 1.0
            for kk in range(2):
                V_col[(j_offset + jj) * p + (k_offset + kk)] = 1.0
            for kk in range(2):
                W_col[(k_offset + kk) * n + i_idx] = 1.0
            U_cols.append(U_col)
            V_cols.append(V_col)
            W_cols.append(W_col)

        # 1x2x2 for k in {2,3}: rank 2
        for jj in range(2):
            U_col = [0.0] * (n * m)
            V_col = [0.0] * (m * p)
            W_col = [0.0] * (p * n)
            U_col[i_idx * m + (j_offset + jj)] = 1.0
            for kk in range(2):
                V_col[(j_offset + jj) * p + (k_offset + 2 + kk)] = 1.0
            for kk in range(2):
                W_col[(k_offset + 2 + kk) * n + i_idx] = 1.0
            U_cols.append(U_col)
            V_cols.append(V_col)
            W_cols.append(W_col)

        # 1x2x1 for k in {4}: rank 2
        # C = a0*B0 + a1*B1
        for jj in range(2):
            U_col = [0.0] * (n * m)
            V_col = [0.0] * (m * p)
            W_col = [0.0] * (p * n)
            U_col[i_idx * m + (j_offset + jj)] = 1.0
            V_col[(j_offset + jj) * p + (k_offset + 4)] = 1.0
            W_col[(k_offset + 4) * n + i_idx] = 1.0
            U_cols.append(U_col)
            V_cols.append(V_col)
            W_cols.append(W_col)

    # For i in {0,1}, j in {0,1}: 2x2x5 block
    add_2x2x5_block(0, 0, 0)
    # For i in {0,1}, j in {2,3}: 2x2x5 block
    add_2x2x5_block(0, 2, 0)
    # For i=2, j in {0,1}: 1x2x5 block
    add_1x2x5_block(2, 0, 0)
    # For i=2, j in {2,3}: 1x2x5 block
    add_1x2x5_block(2, 2, 0)

    # Convert to row-major format
    rank = len(U_cols)
    U_mat = [[U_cols[c][r] for c in range(rank)] for r in range(n * m)]
    V_mat = [[V_cols[c][r] for c in range(rank)] for r in range(m * p)]
    W_mat = [[W_cols[c][r] for c in range(rank)] for r in range(p * n)]

    if verify_fast(U_mat, V_mat, W_mat, n, m, p):
        return {"U": U_mat, "V": V_mat, "W": W_mat}

    # Fallback: schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
