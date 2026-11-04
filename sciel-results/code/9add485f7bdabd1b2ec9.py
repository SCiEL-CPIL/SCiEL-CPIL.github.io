import time
import numpy as np

DIMENSIONS = [4, 4, 7]

def construct():
    start_time = time.time()
    DEADLINE = 130.0

    n, m, p = DIMENSIONS  # 4, 4, 7

    def build_schoolbook():
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
        rank_s = 7
        U_s = [[0] * rank_s for _ in range(4)]
        V_s = [[0] * rank_s for _ in range(4)]
        W_s = [[0] * rank_s for _ in range(4)]
        # P1 = (a11+a22)(b11+b22)
        U_s[0][0] = 1; U_s[3][0] = 1
        V_s[0][0] = 1; V_s[1][0] = 1; V_s[2][0] = 1; V_s[3][0] = 1
        W_s[0][0] = 1; W_s[1][0] = 1; W_s[2][0] = 1; W_s[3][0] = 1
        # P2 = (a21+a22)b11
        U_s[2][1] = 1; U_s[3][1] = 1
        V_s[0][1] = 1
        W_s[2][1] = 1; W_s[3][1] = 1
        # P3 = a11(b12-b22)
        U_s[0][2] = 1
        V_s[1][2] = 1; V_s[3][2] = -1
        W_s[1][2] = 1; W_s[3][2] = 1
        # P4 = a22(b21-b11)
        U_s[3][3] = 1
        V_s[2][3] = 1; V_s[0][3] = -1
        W_s[0][3] = 1; W_s[2][3] = 1
        # P5 = (a11+a12)b22
        U_s[0][4] = 1; U_s[1][4] = 1
        V_s[3][4] = 1
        W_s[0][4] = -1; W_s[1][4] = 1
        # P6 = (a21-a11)(b11+b12)
        U_s[2][5] = 1; U_s[0][5] = -1
        V_s[0][5] = 1; V_s[1][5] = 1
        W_s[3][5] = 1
        # P7 = (a12-a22)(b21+b22)
        U_s[1][6] = 1; U_s[3][6] = -1
        V_s[2][6] = 1; V_s[3][6] = 1
        W_s[0][6] = 1
        return U_s, V_s, W_s

    def build_strassen_4x4x4():
        U_s, V_s, W_s = build_strassen_2x2()
        rank_total = 49
        U_44 = [[0] * rank_total for _ in range(16)]
        V_44 = [[0] * rank_total for _ in range(16)]
        W_44 = [[0] * rank_total for _ in range(16)]
        A_block_mults = [
            [1, 0, 0, 1], [0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 0, 1],
            [1, 1, 0, 0], [-1, 0, 1, 0], [0, 1, 0, -1]
        ]
        B_block_mults = [
            [1, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, -1], [-1, 0, 1, 0],
            [0, 0, 0, 1], [1, 1, 0, 0], [0, 0, 1, 1]
        ]
        C_block_mults = [
            [1, 0, 0, 1], [0, 0, 1, -1], [0, 1, 0, 1], [1, 0, 1, 0],
            [-1, 1, 0, 0], [0, 0, 0, 1], [1, 0, 0, 0]
        ]
        for s in range(7):
            A_mults = A_block_mults[s]
            B_mults = B_block_mults[s]
            C_mults = C_block_mults[s]
            for t in range(7):
                col = s * 7 + t
                for bi in range(2):
                    for bj in range(2):
                        a_mult = A_mults[bi * 2 + bj]
                        if a_mult == 0: continue
                        for ii in range(2):
                            for jj in range(2):
                                u_val = U_s[ii * 2 + jj][t]
                                if u_val == 0: continue
                                U_44[(bi*2+ii)*4 + (bj*2+jj)][col] += a_mult * u_val
                for bj in range(2):
                    for bk in range(2):
                        b_mult = B_mults[bj * 2 + bk]
                        if b_mult == 0: continue
                        for ii in range(2):
                            for jj in range(2):
                                v_val = V_s[ii * 2 + jj][t]
                                if v_val == 0: continue
                                V_44[(bj*2+ii)*4 + (bk*2+jj)][col] += b_mult * v_val
                for ci in range(2):
                    for cj in range(2):
                        c_mult = C_mults[ci * 2 + cj]
                        if c_mult == 0: continue
                        for ii in range(2):
                            for jj in range(2):
                                w_val = W_s[ii * 2 + jj][t]
                                if w_val == 0: continue
                                W_44[(ci*2+ii)*4 + (cj*2+jj)][col] += c_mult * w_val
        return U_44, V_44, W_44

    def build_4x4x2_strassen():
        """
        Build rank-28 decomposition for 4x4 x 4x2 -> 4x2.
        Split into 2x2 blocks. 4 block products of 2x2x2, each rank 7.
        U: 16x28, V: 8x28 (4*2), W: 8x28 (2*4)
        """
        U_s, V_s, W_s = build_strassen_2x2()
        rank_total = 28  # 4 * 7
        U_out = [[0] * rank_total for _ in range(16)]
        V_out = [[0] * rank_total for _ in range(8)]
        W_out = [[0] * rank_total for _ in range(8)]
        
        # 4 block products: (A11*B11, A12*B21, A21*B11, A22*B21)
        # A blocks: (0,0),(0,1),(1,0),(1,1) -> 2x2 each
        # B blocks: (0,0),(1,0) -> 2x2 each (B is 4x2, split rows into 2+2)
        # C blocks: (0,0),(1,0) -> 2x2 each
        
        block_A = [(0,0), (0,1), (1,0), (1,1)]
        block_B = [(0,0), (1,0)]  # B row-blocks
        # For each of 4 A-blocks, multiply with corresponding B-block
        # C_top = A11*B11 + A12*B21
        # C_bot = A21*B11 + A22*B21
        
        prod_configs = [
            # (a_row_block, a_col_block, b_row_block, c_row_block)
            (0, 0, 0, 0),  # A11 * B11 -> C11
            (0, 1, 1, 0),  # A12 * B21 -> C12
            (1, 0, 0, 1),  # A21 * B11 -> C21
            (1, 1, 1, 1),  # A22 * B21 -> C22
        ]
        
        for pi, (ar, ac, br, cr) in enumerate(prod_configs):
            for t in range(7):
                col = pi * 7 + t
                for ii in range(2):
                    for jj in range(2):
                        u_val = U_s[ii * 2 + jj][t]
                        if u_val == 0: continue
                        U_out[(ar*2+ii)*4 + (ac*2+jj)][col] += u_val
                        v_val = V_s[ii * 2 + jj][t]
                        if v_val == 0: continue
                        V_out[(br*2+ii)*2 + (jj if jj < 2 else jj)][col] += v_val
                        # B is 4x2, so row index is 0-3, col is 0-1
                        # B block (br, 0): rows br*2 to br*2+1, cols 0-1
                        # v_row = (br*2+ii)*2 + jj
                        w_val = W_s[ii * 2 + jj][t]
                        if w_val == 0: continue
                        W_out[(cr*2+ii)*4 + (jj)][col] += w_val
                        # W is 2x4 (p*n = 2*4), row = k*4+i
                        # C block (cr, 0): rows cr*2 to cr*2+1, cols 0-3
                        # w_row = (cr*2+ii)*4 + jj... wait, C is 4x2, so p=2, n=4
                        # W index: k*n + i where k is col of C (0-1), i is row of C (0-3)
                        # C row = cr*2+ii, C col = 0 (since we're doing 4x4x2, C has 2 cols)
                        # But in the 2x2x2 subproblem, C is 2x2, so both dims are 2
                        # The outer C is 4x2. C row index = cr*2+ii (0-3), C col index = 0 (only 2 cols total)
                        # Wait, B is 4x2, so C is 4x2. C col index ranges 0-1.
                        # In the block structure, B11 is rows 0-1, cols 0-1 of B.
                        # B21 is rows 2-3, cols 0-1 of B.
                        # So C11 = A11*B11 is rows 0-1, cols 0-1 of C.
                        # C12 = A12*B21 is rows 0-1, cols 0-1 of C. Wait no.
                        # A is 4x4, B is 4x2.
                        # C[i,k] = sum_j A[i,j]*B[j,k]
                        # Split: i = 2*ir + ii, j = 2*jr + jj, k = 2*kr + kk (but k only 0-1, so kr=0, kk=0 or 1)
                        # Actually B has 2 columns, so k=0,1. No block structure in k.
                        # C_top (rows 0-1): C[0:2, 0:2] = A[0:2, 0:4] * B[0:4, 0:2]
                        #   = A11 * B11 + A12 * B21 where A11=A[0:2,0:2], A12=A[0:2,2:4]
                        #   B11=B[0:2,0:2], B21=B[2:4,0:2]
                        # C_bot (rows 2-3): C[2:4, 0:2] = A[2:4, 0:4] * B[0:4, 0:2]
                        #   = A21 * B11 + A22 * B21
                        # So C is 4x2. W index: k*4+i where k=0,1 (col of C), i=0-3 (row of C)
                        # For C block at row-block cr: rows cr*2 to cr*2+1
                        # C col is 0 or 1 (no block)
                        # In the 2x2x2 subproblem, the "k" dimension of C is 2 (cols 0-1 of the 2x2 block)
                        # But here C only has 2 cols total, so the 2x2 C block spans cols 0-1.
                        # w_row = k*4 + i = kk*4 + (cr*2+ii) where kk is 0 or 1
                        # In W_s, index is k*2+i for the 2x2 case. So W_s[ii*2+jj] means k=jj, i=ii
                        # Mapping: outer C row = cr*2+ii, outer C col = jj (0 or 1)
                        # W outer index = jj*4 + (cr*2+ii)
        
        # Let me redo this more carefully
        U_out = [[0] * rank_total for _ in range(16)]
        V_out = [[0] * rank_total for _ in range(8)]
        W_out = [[0] * rank_total for _ in range(8)]
        
        for pi, (ar, ac, br, cr) in enumerate(prod_configs):
            for t in range(7):
                col = pi * 7 + t
                for ii in range(2):
                    for jj in range(2):
                        u_val = U_s[ii * 2 + jj][t]
                        if u_val != 0:
                            U_out[(ar*2+ii)*4 + (ac*2+jj)][col] += u_val
                        v_val = V_s[ii * 2 + jj][t]
                        if v_val != 0:
                            # V_s index: ii*2+jj -> row=ii, col=jj in 2x2 B block
                            # Outer B: row = br*2+ii, col = jj (B is 4x2, cols 0-1)
                            V_out[(br*2+ii)*2 + jj][col] += v_val
                        w_val = W_s[ii * 2 + jj][t]
                        if w_val != 0:
                            # W_s index: ii*2+jj -> for 2x2 C, row=ii, col=jj
                            # But W convention: W[k*n+i] where k=col, i=row
                            # In 2x2x2: W[k*2+i], so index = k*2+i. W_s[ii*2+jj] means k=jj, i=ii
                            # Outer C: row = cr*2+ii, col = jj (C is 4x2, cols 0-1)
                            # W outer index = col*4 + row = jj*4 + (cr*2+ii)
                            W_out[jj*4 + (cr*2+ii)][col] += w_val
        return U_out, V_out, W_out

    def build_4x4x1_schoolbook():
        """
        Build rank-8 decomposition for 4x4 x 4x1 -> 4x1.
        U: 16x8, V: 4x8, W: 4x8
        """
        rank_total = 8
        U_out = [[0] * rank_total for _ in range(16)]
        V_out = [[0] * rank_total for _ in range(4)]
        W_out = [[0] * rank_total for _ in range(4)]
        col = 0
        for i in range(4):
            for j in range(4):
                U_out[i*4+j][col] = 1
                V_out[j*1+0][col] = 1  # B is 4x1, row=j, col=0
                W_out[0*4+i][col] = 1  # C is 4x1, k=0, i=i
                col += 1
        return U_out, V_out, W_out

    def build_4x4x3_optimized():
        """
        Build rank-36 decomposition for 4x4 x 4x3 -> 4x3.
        Split columns: first 2 cols use 4x4x2 Strassen (rank 28), last 1 col uses 4x4x1 (rank 8).
        U: 16x36, V: 12x36 (4*3), W: 12x36 (3*4)
        """
        U2, V2, W2 = build_4x4x2_strassen()
        U1, V1, W1 = build_4x4x1_schoolbook()
        rank2 = 28
        rank1 = 8
        total_rank = rank2 + rank1  # 36
        U_out = [[0] * total_rank for _ in range(16)]
        V_out = [[0] * total_rank for _ in range(12)]  # 4*3
        W_out = [[0] * total_rank for _ in range(12)]  # 3*4
        
        # Copy 4x4x2 part (cols 0-1 of B)
        for r in range(16):
            for c in range(rank2):
                U_out[r][c] = U2[r][c]
        for r in range(8):
            for c in range(rank2):
                V_out[r][c] = V2[r][c]
        for r in range(8):
            for c in range(rank2):
                W_out[r][c] = W2[r][c]
        
        # Copy 4x4x1 part (col 2 of B)
        off = rank2
        for r in range(16):
            for c in range(rank1):
                U_out[r][off+c] = U1[r][c]
        for r in range(4):
            for c in range(rank1):
                V_out[8+r][off+c] = V1[r][c]  # B row offset: col 2 starts at V row 8
        for r in range(4):
            for c in range(rank1):
                W_out[8+r][off+c] = W1[r][c]  # W row offset: col 2 starts at W row 8
        
        return U_out, V_out, W_out

    def verify_decomposition(U, V, W, n=4, m=4, p=7):
        rank = len(U[0])
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        tensor_decomp = np.zeros((n, m, p))
        for c in range(rank):
            u_col = U_arr[:, c]
            v_col = V_arr[:, c]
            w_col = W_arr[:, c]
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        tensor_decomp[i, j, k] += u_col[i*m+j] * v_col[j*p+k] * w_col[k*n+i]
        if not np.allclose(tensor_decomp, 1.0, atol=1e-9):
            return False
        return True

    def build_4x4x7_via_strassen():
        """
        Build rank-85 decomposition for 4x4 x 4x7:
        - First 4 columns: Strassen 4x4x4 (rank 49)
        - Next 2 columns: 4x4x2 Strassen (rank 28)
        - Last 1 column: 4x4x1 schoolbook (rank 8)
        Total: 85
        """
        U_44, V_44, W_44 = build_strassen_4x4x4()
        U_42, V_42, W_42 = build_4x4x2_strassen()
        U_41, V_41, W_41 = build_4x4x1_schoolbook()
        
        r44 = 49
        r42 = 28
        r41 = 8
        total_rank = r44 + r42 + r41  # 85
        
        U_out = [[0] * total_rank for _ in range(16)]
        V_out = [[0] * total_rank for _ in range(28)]  # 4*7
        W_out = [[0] * total_rank for _ in range(28)]  # 7*4
        
        # 4x4x4 part: B cols 0-3, W rows 0-15 (k=0-3, i=0-3)
        for r in range(16):
            for c in range(r44):
                U_out[r][c] = U_44[r][c]
        for r in range(16):
            for c in range(r44):
                V_out[r][c] = V_44[r][c]
        for r in range(16):
            for c in range(r44):
                W_out[r][c] = W_44[r][c]
        
        # 4x4x2 part: B cols 4-5, W rows 16-23 (k=4-5, i=0-3)
        off = r44
        for r in range(16):
            for c in range(r42):
                U_out[r][off+c] = U_42[r][c]
        for r in range(8):
            for c in range(r42):
                V_out[16+r][off+c] = V_42[r][c]  # B rows offset: cols 4-5 start at V row 16
        for r in range(8):
            for c in range(r42):
                W_out[16+r][off+c] = W_42[r][c]  # W rows offset: k=4-5 start at W row 16
        
        # 4x4x1 part: B col 6, W row 24-27 (k=6, i=0-3)
        off2 = r44 + r42
        for r in range(16):
            for c in range(r41):
                U_out[r][off2+c] = U_41[r][c]
        for r in range(4):
            for c in range(r41):
                V_out[24+r][off2+c] = V_41[r][c]  # B row offset: col 6 starts at V row 24
        for r in range(4):
            for c in range(r41):
                W_out[24+r][off2+c] = W_41[r][c]  # W row offset: k=6 starts at W row 24
        
        return {"U": U_out, "V": V_out, "W": W_out}

    if time.time() - start_time > DEADLINE:
        return build_schoolbook()

    try:
        result = build_4x4x7_via_strassen()
        if not verify_decomposition(result["U"], result["V"], result["W"]):
            return build_schoolbook()
        if time.time() - start_time > DEADLINE:
            return build_schoolbook()
        return result
    except Exception:
        return build_schoolbook()
