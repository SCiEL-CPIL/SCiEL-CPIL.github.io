import time
import numpy as np

DIMENSIONS = [4, 4, 7]

def construct():
    start_time = time.time()
    DEADLINE = 120.0  # safety margin before 150s kill

    n, m, p = DIMENSIONS  # 4, 4, 7
    rank_schoolbook = n * m * p  # 112

    def build_schoolbook():
        """Return the schoolbook (trivial) decomposition of rank n*m*p."""
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

    def build_strassen_2x2():
        """
        Return the rank-7 Strassen decomposition for 2x2 x 2x2 -> 2x2.
        U: 4x7 (rows indexed by (i,j) of A), V: 4x7 (rows indexed by (j,k) of B), W: 4x7 (rows indexed by (k,i) of C)
        Using the trace convention: W[k*n + i, c] for n=2.
        """
        rank_s = 7
        U_s = [[0] * rank_s for _ in range(4)]
        V_s = [[0] * rank_s for _ in range(4)]
        W_s = [[0] * rank_s for _ in range(4)]

        # P1 = (a11+a22)(b11+b22)
        U_s[0][0] = 1  # a11
        U_s[3][0] = 1  # a22
        V_s[0][0] = 1  # b11
        V_s[1][0] = 1  # b12
        V_s[2][0] = 1  # b21
        V_s[3][0] = 1  # b22
        W_s[0][0] = 1  # c11 (k=0,i=0)
        W_s[1][0] = 1  # c12 (k=0,i=1)
        W_s[2][0] = 1  # c21 (k=1,i=0)
        W_s[3][0] = 1  # c22 (k=1,i=1)

        # P2 = (a21+a22)b11
        U_s[2][1] = 1  # a21
        U_s[3][1] = 1  # a22
        V_s[0][1] = 1  # b11
        W_s[2][1] = 1  # c21 (k=1,i=0)
        W_s[3][1] = 1  # c22 (k=1,i=1)

        # P3 = a11(b12-b22)
        U_s[0][2] = 1  # a11
        V_s[1][2] = 1  # b12
        V_s[3][2] = -1  # b22
        W_s[1][2] = 1  # c12 (k=0,i=1)
        W_s[3][2] = 1  # c22 (k=1,i=1)

        # P4 = a22(b21-b11)
        U_s[3][3] = 1  # a22
        V_s[2][3] = 1  # b21
        V_s[0][3] = -1  # b11
        W_s[0][3] = 1  # c11 (k=0,i=0)
        W_s[2][3] = 1  # c21 (k=1,i=0)

        # P5 = (a11+a12)b22
        U_s[0][4] = 1  # a11
        U_s[1][4] = 1  # a12
        V_s[3][4] = 1  # b22
        W_s[0][4] = -1  # c11 (k=0,i=0)
        W_s[1][4] = 1  # c12 (k=0,i=1)

        # P6 = (a21-a11)(b11+b12)
        U_s[2][5] = 1  # a21
        U_s[0][5] = -1  # a11
        V_s[0][5] = 1  # b11
        V_s[1][5] = 1  # b12
        W_s[3][5] = 1  # c22 (k=1,i=1)

        # P7 = (a12-a22)(b21+b22)
        U_s[1][6] = 1  # a12
        U_s[3][6] = -1  # a22
        V_s[2][6] = 1  # b21
        V_s[3][6] = 1  # b22
        W_s[0][6] = 1  # c11 (k=0,i=0)

        return U_s, V_s, W_s

    def build_strassen_4x4x4():
        """
        Build a rank-49 decomposition for 4x4 x 4x4 -> 4x4 using recursive Strassen.
        Split into 2x2 blocks. Use 7 Strassen products, each computed with rank-7 Strassen.
        U: 16x49, V: 16x49, W: 16x49 (W indexed by k*4+i)
        """
        U_s, V_s, W_s = build_strassen_2x2()
        rank_s = 7
        rank_total = 49  # 7 * 7

        U_44 = [[0] * rank_total for _ in range(16)]
        V_44 = [[0] * rank_total for _ in range(16)]
        W_44 = [[0] * rank_total for _ in range(16)]

        # Strassen block structure for 4x4:
        A_block_mults = [
            [1, 0, 0, 1],   # P1: A_11+A_22
            [0, 0, 1, 1],   # P2: A_21+A_22
            [1, 0, 0, 0],   # P3: A_11
            [0, 0, 0, 1],   # P4: A_22
            [1, 1, 0, 0],   # P5: A_11+A_12
            [-1, 0, 1, 0],  # P6: A_21-A_11
            [0, 1, 0, -1],  # P7: A_12-A_22
        ]

        B_block_mults = [
            [1, 0, 0, 1],   # P1: B_11+B_22
            [1, 0, 0, 0],   # P2: B_11
            [0, 1, 0, -1],  # P3: B_12-B_22
            [-1, 0, 1, 0],  # P4: B_21-B_11
            [0, 0, 0, 1],   # P5: B_22
            [1, 1, 0, 0],   # P6: B_11+B_12
            [0, 0, 1, 1],   # P7: B_21+B_22
        ]

        C_block_mults = [
            [1, 0, 0, 1],   # P1: C_11=1, C_22=1
            [0, 0, 1, -1],  # P2: C_21=1, C_22=-1
            [0, 1, 0, 1],   # P3: C_12=1, C_22=1
            [1, 0, 1, 0],   # P4: C_11=1, C_21=1
            [-1, 1, 0, 0],  # P5: C_11=-1, C_12=1
            [0, 0, 0, 1],   # P6: C_22=1
            [1, 0, 0, 0],   # P7: C_11=1
        ]

        for s in range(7):
            A_mults = A_block_mults[s]
            B_mults = B_block_mults[s]
            C_mults = C_block_mults[s]

            for t in range(7):
                col = s * 7 + t

                # For each entry in the 2x2 A-block (inner indices ii, jj):
                for bi in range(2):
                    for bj in range(2):
                        a_mult = A_mults[bi * 2 + bj]
                        if a_mult == 0:
                            continue
                        for ii in range(2):
                            for jj in range(2):
                                u_val = U_s[ii * 2 + jj][t]
                                if u_val == 0:
                                    continue
                                a4_i = bi * 2 + ii
                                a4_j = bj * 2 + jj
                                u_row = a4_i * 4 + a4_j
                                U_44[u_row][col] += a_mult * u_val

                # For each entry in the 2x2 B-block:
                for bj in range(2):
                    for bk in range(2):
                        b_mult = B_mults[bj * 2 + bk]
                        if b_mult == 0:
                            continue
                        for ii in range(2):
                            for jj in range(2):
                                v_val = V_s[ii * 2 + jj][t]
                                if v_val == 0:
                                    continue
                                b4_j = bj * 2 + ii  # row of B (shared dim)
                                b4_k = bk * 2 + jj  # col of B
                                v_row = b4_j * 4 + b4_k
                                V_44[v_row][col] += b_mult * v_val

                # For each entry in the 2x2 C-block:
                for ci in range(2):
                    for cj in range(2):
                        c_mult = C_mults[ci * 2 + cj]
                        if c_mult == 0:
                            continue
                        for ii in range(2):
                            for jj in range(2):
                                w_val = W_s[ii * 2 + jj][t]
                                if w_val == 0:
                                    continue
                                c4_k = ci * 2 + ii  # row of C result index
                                c4_i = cj * 2 + jj  # col of C result index
                                w_row = c4_k * 4 + c4_i
                                W_44[w_row][col] += c_mult * w_val

        return U_44, V_44, W_44

    def verify_decomposition(U, V, W, n=4, m=4, p=7):
        """Verify that the decomposition exactly matches the matrix multiplication tensor."""
        rank = len(U[0])
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        # The multiplication tensor T[i,j,k] should equal 1 for all valid (i,j,k)
        # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        
        tensor_decomp = np.zeros((n, m, p))
        for c in range(rank):
            u_col = U_arr[:, c]
            v_col = V_arr[:, c]
            w_col = W_arr[:, c]
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        tensor_decomp[i, j, k] += u_col[i * m + j] * v_col[j * p + k] * w_col[k * n + i]
        
        if not np.allclose(tensor_decomp, 1.0, atol=1e-9):
            return False
        return True

    def build_4x4x7_via_strassen():
        """
        Build a rank-97 decomposition for 4x4 x 4x7:
        - First 4 columns of B: use Strassen 4x4x4 (rank 49)
        - Last 3 columns of B: use schoolbook 4x4x3 (rank 48)
        Total: 97
        """
        U_44, V_44, W_44 = build_strassen_4x4x4()

        rank_strassen = 49
        rank_schoolbook_3 = 4 * 4 * 3  # 48
        total_rank = rank_strassen + rank_schoolbook_3  # 97

        U_out = [[0] * total_rank for _ in range(16)]
        V_out = [[0] * total_rank for _ in range(28)]  # m*p = 4*7 = 28
        W_out = [[0] * total_rank for _ in range(28)]  # p*n = 7*4 = 28

        # Copy Strassen part (first 4 columns of B, first 4 rows of W)
        for r in range(16):
            for c in range(rank_strassen):
                U_out[r][c] = U_44[r][c]
        for r in range(16):
            for c in range(rank_strassen):
                V_out[r][c] = V_44[r][c]
        for r in range(16):
            for c in range(rank_strassen):
                W_out[r][c] = W_44[r][c]

        # Schoolbook for columns 4,5,6 of B (0-indexed: cols 4,5,6)
        col_offset = rank_strassen
        for i in range(4):
            for j in range(4):
                for k in range(3):
                    kk = k + 4  # actual column index in B
                    U_out[i * 4 + j][col_offset] = 1
                    V_out[j * 7 + kk][col_offset] = 1
                    W_out[kk * 4 + i][col_offset] = 1
                    col_offset += 1

        return {"U": U_out, "V": V_out, "W": W_out}

    # Check time
    if time.time() - start_time > DEADLINE:
        return build_schoolbook()

    # Try the Strassen-based decomposition
    try:
        result = build_4x4x7_via_strassen()
        
        # Verify the decomposition
        U = result["U"]
        V = result["V"]
        W = result["W"]
        
        if not verify_decomposition(U, V, W):
            # Decomposition is incorrect, fall back to schoolbook
            return build_schoolbook()

        if time.time() - start_time > DEADLINE:
            return build_schoolbook()

        return result

    except Exception:
        return build_schoolbook()
