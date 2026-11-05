"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 4, 5
    deadline = time.time() + 110.0

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

    def strassen_2x2x2():
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
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        rank = U.shape[1]
        U_r = U.reshape(n, m, rank)
        V_r = V.reshape(m, p, rank)
        W_r = W.reshape(p, n, rank)
        T = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r, optimize=True)
        return np.allclose(T, np.ones((n, m, p)), atol=tol, rtol=0)

    def greedy_reduce(U_mat, V_mat, W_mat, n, m, p, tol=1e-9):
        """Try to remove columns one by one if the decomposition still holds."""
        U = [list(row) for row in U_mat]
        V = [list(row) for row in V_mat]
        W = [list(row) for row in W_mat]
        rank = len(U[0])
        
        # Try removing each column in order
        removed = []
        for c in range(rank):
            if time.time() > deadline - 5:
                break
            # Skip if already removed
            if c in removed:
                continue
            # Try removing column c
            U_test = [row[:c] + row[c+1:] for row in U]
            V_test = [row[:c] + row[c+1:] for row in V]
            W_test = [row[:c] + row[c+1:] for row in W]
            if verify_fast(U_test, V_test, W_test, n, m, p, tol):
                U = U_test
                V = V_test
                W = W_test
                removed.append(c)
                rank -= 1
        
        return U, V, W, rank

    def build_strassen_blocks():
        """Build the 4-block Strassen-based decomposition (rank 56)."""
        U_cols = []
        V_cols = []
        W_cols = []

        def add_2x2x2_strassen(i_start, j_start, k_start):
            U_s, V_s, W_s = strassen_2x2x2()
            for c in range(7):
                U_col = [0.0] * (n * m)
                V_col = [0.0] * (m * p)
                W_col = [0.0] * (p * n)
                for i in range(2):
                    for j in range(2):
                        U_col[(i_start+i)*m + (j_start+j)] = U_s[i*2+j][c]
                for j in range(2):
                    for k in range(2):
                        V_col[(j_start+j)*p + (k_start+k)] = V_s[j*2+k][c]
                for k in range(2):
                    for i in range(2):
                        W_col[(k_start+k)*n + (i_start+i)] = W_s[k*2+i][c]
                U_cols.append(U_col)
                V_cols.append(V_col)
                W_cols.append(W_col)

        def add_2x2x1(i_start, j_start, k_pos):
            U_col = [0.0] * (n * m)
            V_col = [0.0] * (m * p)
            W_col = [0.0] * (p * n)
            for i in range(2):
                for j in range(2):
                    U_col[(i_start+i)*m + (j_start+j)] = 1.0
            for j in range(2):
                V_col[(j_start+j)*p + k_pos] = 1.0
            for i in range(2):
                W_col[k_pos*n + (i_start+i)] = 1.0
            U_cols.append(U_col)
            V_cols.append(V_col)
            W_cols.append(W_col)

        # Block 1: i in {0,1}, j in {0,1}
        add_2x2x2_strassen(0, 0, 0)
        add_2x2x2_strassen(0, 0, 2)
        add_2x2x1(0, 0, 4)

        # Block 2: i in {0,1}, j in {2,3}
        add_2x2x2_strassen(0, 2, 0)
        add_2x2x2_strassen(0, 2, 2)
        add_2x2x1(0, 2, 4)

        # Block 3: i=2, j in {0,1} -> rank 1
        U_col = [0.0] * (n * m)
        V_col = [0.0] * (m * p)
        W_col = [0.0] * (p * n)
        U_col[2*m + 0] = 1.0
        U_col[2*m + 1] = 1.0
        for j in range(2):
            for k in range(5):
                V_col[j*p + k] = 1.0
        for k in range(5):
            W_col[k*n + 2] = 1.0
        U_cols.append(U_col)
        V_cols.append(V_col)
        W_cols.append(W_col)

        # Block 4: i=2, j in {2,3} -> rank 1
        U_col = [0.0] * (n * m)
        V_col = [0.0] * (m * p)
        W_col = [0.0] * (p * n)
        U_col[2*m + 2] = 1.0
        U_col[2*m + 3] = 1.0
        for j in range(2, 4):
            for k in range(5):
                V_col[j*p + k] = 1.0
        for k in range(5):
            W_col[k*n + 2] = 1.0
        U_cols.append(U_col)
        V_cols.append(V_col)
        W_cols.append(W_col)

        rank = len(U_cols)
        U_mat = [[U_cols[c][r] for c in range(rank)] for r in range(n*m)]
        V_mat = [[V_cols[c][r] for c in range(rank)] for r in range(m*p)]
        W_mat = [[W_cols[c][r] for c in range(rank)] for r in range(p*n)]
        return U_mat, V_mat, W_mat

    def build_2x3x5_blocks():
        """Try 2x3x5 blocks for first two rows of A."""
        U_cols = []
        V_cols = []
        W_cols = []

        # For i in {0,1}, j in {0,1,2}: use 2x3x5
        # Split k into {0,1,2} and {3,4}
        # k in {0,1,2}: 2x3x3
        # k in {3,4}: 2x3x2

        # 2x3x3 has known rank 12 (schoolbook) or better
        # 2x3x2 has known rank 11 or so

        # Let's just use schoolbook for these sub-blocks for now
        # i in {0,1}, j in {0,1,2}, k in {0,1,2}: 2x3x3 = 18 rank
        for i in range(2):
            for j in range(3):
                for k in range(3):
                    U_col = [0.0] * (n * m)
                    V_col = [0.0] * (m * p)
                    W_col = [0.0] * (p * n)
                    U_col[i*m + j] = 1.0
                    V_col[j*p + k] = 1.0
                    W_col[k*n + i] = 1.0
                    U_cols.append(U_col)
                    V_cols.append(V_col)
                    W_cols.append(W_col)

        # i in {0,1}, j in {0,1,2}, k in {3,4}: 2x3x2 = 12 rank
        for i in range(2):
            for j in range(3):
                for k in range(2):
                    U_col = [0.0] * (n * m)
                    V_col = [0.0] * (m * p)
                    W_col = [0.0] * (p * n)
                    U_col[i*m + j] = 1.0
                    V_col[j*p + (3+k)] = 1.0
                    W_col[(3+k)*n + i] = 1.0
                    U_cols.append(U_col)
                    V_cols.append(V_col)
                    W_cols.append(W_col)

        # i in {0,1}, j=3, k in {0,1,2,3,4}: 2x1x5 = 10 rank
        for i in range(2):
            for k in range(5):
                U_col = [0.0] * (n * m)
                V_col = [0.0] * (m * p)
                W_col = [0.0] * (p * n)
                U_col[i*m + 3] = 1.0
                V_col[3*p + k] = 1.0
                W_col[k*n + i] = 1.0
                U_cols.append(U_col)
                V_cols.append(V_col)
                W_cols.append(W_col)

        # i=2, j in {0,1,2,3}, k in {0,1,2,3,4}: 1x4x5 = 20 rank
        for j in range(4):
            for k in range(5):
                U_col = [0.0] * (n * m)
                V_col = [0.0] * (m * p)
                W_col = [0.0] * (p * n)
                U_col[2*m + j] = 1.0
                V_col[j*p + k] = 1.0
                W_col[k*n + 2] = 1.0
                U_cols.append(U_col)
                V_cols.append(V_col)
                W_cols.append(W_col)

        rank = len(U_cols)
        U_mat = [[U_cols[c][r] for c in range(rank)] for r in range(n*m)]
        V_mat = [[V_cols[c][r] for c in range(rank)] for r in range(m*p)]
        W_mat = [[W_cols[c][r] for c in range(rank)] for r in range(p*n)]
        return U_mat, V_mat, W_mat

    best_result = schoolbook()
    best_rank = n * m * p

    # Try the Strassen-based construction
    U_mat, V_mat, W_mat = build_strassen_blocks()
    if verify_fast(U_mat, V_mat, W_mat, n, m, p):
        # Try greedy reduction
        U_red, V_red, W_red, rank_red = greedy_reduce(U_mat, V_mat, W_mat, n, m, p)
        if rank_red < best_rank:
            best_result = {"U": U_red, "V": V_red, "W": W_red}
            best_rank = rank_red
        elif len(U_mat[0]) < best_rank:
            best_result = {"U": U_mat, "V": V_mat, "W": W_mat}
            best_rank = len(U_mat[0])

    # Try alternative block decomposition
    if time.time() < deadline - 10:
        U_mat2, V_mat2, W_mat2 = build_2x3x5_blocks()
        if verify_fast(U_mat2, V_mat2, W_mat2, n, m, p):
            U_red2, V_red2, W_red2, rank_red2 = greedy_reduce(U_mat2, V_mat2, W_mat2, n, m, p)
            if rank_red2 < best_rank:
                best_result = {"U": U_red2, "V": V_red2, "W": W_red2}
                best_rank = rank_red2

    return best_result
# EVOLVE-BLOCK-END
