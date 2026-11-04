import numpy as np
import time
from fractions import Fraction

def construct():
    n, m, p = 2, 5, 6
    deadline = time.time() + 140

    # Strassen's 7 bilinear forms for 2x2 matrix multiplication:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B21 - B22)
    # M4 = A22(B12 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    #
    # For our 2x5 * 5x6:
    # A is 2x5: A11 (row0, cols0:2), A12 (row0, cols2:5), A21 (row1, cols0:2), A22 (row1, cols2:5)
    # B is 5x6: B11 (rows0:2, cols0:2), B12 (rows0:2, cols2:5), B21 (rows2:5, cols0:2), B22 (rows2:5, cols2:5)
    #
    # We split B's columns into 3 blocks of width 2: B[:, 0:2], B[:, 2:4], B[:, 4:6]
    # For each column block, we apply Strassen to the 2x2 block structure.
    #
    # Actually, let me think more carefully. The standard approach:
    # Split A into 2x2 blocks where each "element" is a 2x3 or 3x3 submatrix? No.
    #
    # Better: Think of it as multiplying 2x5 by 5x6.
    # Split the 5 intermediate dimension into 2+3.
    # A = [A_left (2x2), A_right (2x3)]
    # B = [B_top (2x6), B_bottom (3x6)]
    # C = A_left * B_top + A_right * B_bottom
    #
    # For A_left (2x2) * B_top (2x6): split B_top columns into 3 blocks of 2.
    #   For each block of 2 columns, we have a 2x2 * 2x2 multiplication.
    #   Use Strassen: 7 products per block, 3 blocks = 21 products.
    #
    # For A_right (2x3) * B_bottom (3x6): this is harder.
    #   We can split into: A_right[:, 0:2] * B_bottom[0:2, :] + A_right[:, 2:3] * B_bottom[2:3, :]
    #   A_right[:, 0:2] is 2x2, B_bottom[0:2, :] is 2x6. Same as above: 21 products.
    #   A_right[:, 2:3] is 2x1, B_bottom[2:3, :] is 1x6. This is 2*1*6 = 12 products (schoolbook).
    #
    # Total: 21 + 21 + 12 = 54. That matches the known result!
    #
    # Let me construct this explicitly.
    #
    # Indices:
    # A[i, j] where i in {0,1}, j in {0,1,2,3,4}
    # B[j, k] where j in {0,1,2,3,4}, k in {0,1,2,3,4,5}
    # C[i, k] where i in {0,1}, k in {0,1,2,3,4,5}
    #
    # Tensor convention: W[k*n + i, r] = W[k*2 + i, r]
    # U[i*m + j, r] = U[i*5 + j, r]
    # V[j*p + k, r] = V[j*6 + k, r]
    #
    # Part 1: A_left (2x2) * B_top (2x6), where A_left = A[:, 0:2], B_top = B[0:2, :]
    #   Split B_top into 3 column blocks of width 2:
    #   Block c (c=0,1,2): B_top[:, c*2 : c*2+2] is 2x2
    #   For block c, Strassen gives 7 products.
    #
    # Part 2: A_right_top (2x2) * B_mid (2x6), where A_right_top = A[:, 2:4], B_mid = B[2:4, :]
    #   Same structure: 7 products per column block, 3 blocks = 21 products.
    #
    # Part 3: A_right_bottom (2x1) * B_bottom (1x6), where A_right_bottom = A[:, 4:5], B_bottom = B[4:5, :]
    #   Schoolbook: 2*1*6 = 12 products.
    #
    # Total rank = 21 + 21 + 12 = 54.

    # Let's build the factor matrices.
    # Each "product" in Strassen is a bilinear form: (linear combo of A entries) * (linear combo of B entries)
    # For each such product, we need one column in U, V, W.

    rank = 54
    U = [[0] * rank for _ in range(n * m)]  # 10 x 54
    V = [[0] * rank for _ in range(m * p)]  # 30 x 54
    W = [[0] * rank for _ in range(p * n)]  # 12 x 54

    col = 0

    # Helper: add a Strassen decomposition for a 2x2 * 2x2 block
    # A block: rows i1,i2 of A, cols j1,j2
    # B block: rows j1,j2 of B, cols k1,k2
    # C output: rows i1,i2 of C, cols k1,k2
    def add_strassen(i1, i2, j1, j2, k1, k2):
        nonlocal col
        # M1 = (A[i1,j1] + A[i2,j2]) * (B[j1,k1] + B[j2,k2])
        # M2 = (A[i2,j1] + A[i2,j2]) * B[j1,k1]
        # M3 = A[i1,j1] * (B[j2,k1] - B[j2,k2])
        # M4 = A[i2,j2] * (B[j1,k2] - B[j1,k1])
        # M5 = (A[i1,j1] + A[i1,j2]) * B[j2,k2]
        # M6 = (A[i2,j1] - A[i1,j1]) * (B[j1,k1] + B[j1,k2])
        # M7 = (A[i1,j2] - A[i2,j2]) * (B[j2,k1] + B[j2,k2])
        #
        # C[i1,k1] += M1 + M4 - M5 + M7
        # C[i1,k2] += M3 + M5
        # C[i2,k1] += M2 + M4
        # C[i2,k2] += M1 - M2 + M3 + M6

        # M1
        U[i1 * m + j1][col] += 1
        U[i2 * m + j2][col] += 1
        V[j1 * p + k1][col] += 1
        V[j2 * p + k2][col] += 1
        W[k1 * n + i1][col] += 1
        W[k2 * n + i2][col] += 1
        col += 1

        # M2
        U[i2 * m + j1][col] += 1
        U[i2 * m + j2][col] += 1
        V[j1 * p + k1][col] += 1
        W[k1 * n + i2][col] += 1
        W[k2 * n + i2][col] += 1
        col += 1

        # M3
        U[i1 * m + j1][col] += 1
        V[j2 * p + k1][col] += 1
        V[j2 * p + k2][col] += -1
        W[k1 * n + i1][col] += 1
        W[k2 * n + i1][col] += 1
        col += 1

        # M4
        U[i2 * m + j2][col] += 1
        V[j1 * p + k2][col] += 1
        V[j1 * p + k1][col] += -1
        W[k1 * n + i1][col] += 1
        W[k2 * n + i2][col] += 1
        col += 1

        # M5
        U[i1 * m + j1][col] += 1
        U[i1 * m + j2][col] += 1
        V[j2 * p + k2][col] += 1
        W[k1 * n + i1][col] += -1
        W[k2 * n + i1][col] += 1
        col += 1

        # M6
        U[i2 * m + j1][col] += 1
        U[i1 * m + j1][col] += -1
        V[j1 * p + k1][col] += 1
        V[j1 * p + k2][col] += 1
        W[k2 * n + i2][col] += 1
        col += 1

        # M7
        U[i1 * m + j2][col] += 1
        U[i2 * m + j2][col] += -1
        V[j2 * p + k1][col] += 1
        V[j2 * p + k2][col] += 1
        W[k1 * n + i1][col] += 1
        W[k2 * n + i2][col] += -1
        col += 1

    # Part 1: A[:, 0:2] * B[0:2, :]
    # A rows: 0,1; A cols: 0,1
    # B rows: 0,1; B cols: 0,1,2,3,4,5
    # Split B cols into blocks of 2: (0,1), (2,3), (4,5)
    for c in range(3):
        k1 = c * 2
        k2 = c * 2 + 1
        add_strassen(0, 1, 0, 1, k1, k2)

    # Part 2: A[:, 2:4] * B[2:4, :]
    # A rows: 0,1; A cols: 2,3
    # B rows: 2,3; B cols: 0,1,2,3,4,5
    for c in range(3):
        k1 = c * 2
        k2 = c * 2 + 1
        add_strassen(0, 1, 2, 3, k1, k2)

    # Part 3: A[:, 4:5] * B[4:5, :]
    # A rows: 0,1; A cols: 4
    # B rows: 4; B cols: 0,1,2,3,4,5
    # Schoolbook: for each (i, j=4, k), one product
    for i in range(2):
        for k in range(6):
            U[i * m + 4][col] += 1
            V[4 * p + k][col] += 1
            W[k * n + i][col] += 1
            col += 1

    assert col == rank, f"Expected {rank} columns, got {col}"

    # Verify
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)

    # Check diagonal terms
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                s = np.dot(U_np[a], V_np[b] * W_np[c])
                if abs(s - 1.0) > 1e-9:
                    # Fallback to schoolbook
                    rank_sb = n * m * p
                    U_sb = [[0] * rank_sb for _ in range(n * m)]
                    V_sb = [[0] * rank_sb for _ in range(m * p)]
                    W_sb = [[0] * rank_sb for _ in range(p * n)]
                    col_sb = 0
                    for ii in range(n):
                        for jj in range(m):
                            for kk in range(p):
                                U_sb[ii * m + jj][col_sb] = 1
                                V_sb[jj * p + kk][col_sb] = 1
                                W_sb[kk * n + ii][col_sb] = 1
                                col_sb += 1
                    return {"U": U_sb, "V": V_sb, "W": W_sb}

    # Check some non-diagonal terms
    np.random.seed(42)
    for _ in range(500):
        a = np.random.randint(0, n * m)
        b = np.random.randint(0, m * p)
        c = np.random.randint(0, p * n)
        i_a, j_a = divmod(a, m)
        j_b, k_b = divmod(b, p)
        k_c, i_c = divmod(c, n)
        if not (j_a == j_b and k_b == k_c and i_a == i_c):
            s = np.dot(U_np[a], V_np[b] * W_np[c])
            if abs(s) > 1e-9:
                rank_sb = n * m * p
                U_sb = [[0] * rank_sb for _ in range(n * m)]
                V_sb = [[0] * rank_sb for _ in range(m * p)]
                W_sb = [[0] * rank_sb for _ in range(p * n)]
                col_sb = 0
                for ii in range(n):
                    for jj in range(m):
                        for kk in range(p):
                            U_sb[ii * m + jj][col_sb] = 1
                            V_sb[jj * p + kk][col_sb] = 1
                            W_sb[kk * n + ii][col_sb] = 1
                            col_sb += 1
                return {"U": U_sb, "V": V_sb, "W": W_sb}

    return {"U": U, "V": V, "W": W}
