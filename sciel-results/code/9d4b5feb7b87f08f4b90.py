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
        U_s, V_s, W_s = build_strassen_2x2()
        rank_total = 28
        U_out = [[0] * rank_total for _ in range(16)]
        V_out = [[0] * rank_total for _ in range(8)]
        W_out = [[0] * rank_total for _ in range(8)]
        
        configs = [
            # (a_row, a_col, b_row, c_row, offset)
            (0, 0, 0, 0, 0),   # A11 B11 -> C11
            (0, 2, 2, 0, 7),   # A12 B21 -> C11
            (2, 0, 0, 2, 14),  # A21 B11 -> C21
            (2, 2, 2, 2, 21),  # A22 B21 -> C21
        ]
        
        for ar, ac, br, cr, off in configs:
            for t in range(7):
                col = off + t
                for ii in range(2):
                    for jj in range(2):
                        u_val = U_s[ii * 2 + jj][t]
                        if u_val != 0:
                            U_out[(ar+ii)*4 + (ac+jj)][col] += u_val
                        v_val = V_s[ii * 2 + jj][t]
                        if v_val != 0:
                            V_out[(br+ii)*2 + jj][col] += v_val
                        w_val = W_s[ii * 2 + jj][t]
                        if w_val != 0:
                            W_out[jj*4 + (cr+ii)][col] += w_val
        return U_out, V_out, W_out

    def build_4x4x1_schoolbook():
        rank_total = 16
        U_out = [[0] * rank_total for _ in range(16)]
        V_out = [[0] * rank_total for _ in range(4)]
        W_out = [[0] * rank_total for _ in range(4)]
        col = 0
        for i in range(4):
            for j in range(4):
                U_out[i*4+j][col] = 1
                V_out[j][col] = 1
                W_out[i][col] = 1
                col += 1
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
        U_44, V_44, W_44 = build_strassen_4x4x4()
        U_42, V_42, W_42 = build_4x4x2_strassen()
        U_41, V_41, W_41 = build_4x4x1_schoolbook()
        
        r44 = 49
        r42 = 28
        r41 = 16
        total_rank = r44 + r42 + r41
        
        U_out = [[0] * total_rank for _ in range(16)]
        V_out = [[0] * total_rank for _ in range(28)]
        W_out = [[0] * total_rank for _ in range(28)]
        
        # 4x4x4
        for r in range(16):
            for c in range(r44):
                U_out[r][c] = U_44[r][c]
                V_out[r][c] = V_44[r][c]
                W_out[r][c] = W_44[r][c]
        
        # 4x4x2
        off = r44
        for r in range(16):
            for c in range(r42):
                U_out[r][off+c] = U_42[r][c]
        for r in range(8):
            for c in range(r42):
                # Map V_42 row r to V_out
                j_sub = r // 2
                k_sub = r % 2
                v_row = j_sub * 7 + k_sub + 4
                V_out[v_row][off+c] = V_42[r][c]
        for r in range(8):
            for c in range(r42):
                # Map W_42 row r to W_out
                k_sub = r // 4
                i_sub = r % 4
                w_row = (k_sub + 4) * 4 + i_sub
                W_out[w_row][off+c] = W_42[r][c]
        
        # 4x4x1
        off2 = r44 + r42
        for r in range(16):
            for c in range(r41):
                U_out[r][off2+c] = U_41[r][c]
        for r in range(4):
            for c in range(r41):
                v_row = r * 7 + 6
                V_out[v_row][off2+c] = V_41[r][c]
        for r in range(4):
            for c in range(r41):
                w_row = 6 * 4 + r
                W_out[w_row][off2+c] = W_41[r][c]
        
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
