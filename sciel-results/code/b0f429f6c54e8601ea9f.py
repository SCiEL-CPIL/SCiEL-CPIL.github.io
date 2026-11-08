"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 5
    
    # Deadline guard: return a valid decomposition before timeout
    start_time = time.time()
    DEADLINE = 145  # seconds, with 5s safety margin
    
    # Fallback: Schoolbook decomposition (Rank 80)
    def get_schoolbook():
        rank_fb = n * m * p
        U_fb = [[0] * rank_fb for _ in range(n * m)]
        V_fb = [[0] * rank_fb for _ in range(m * p)]
        W_fb = [[0] * rank_fb for _ in range(p * n)]
        col_fb = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j][col_fb] = 1
                    V_fb[j * p + k][col_fb] = 1
                    W_fb[k * n + i][col_fb] = 1
                    col_fb += 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}

    def verify(U_mat, V_mat, W_mat, rank_val):
        """Verify decomposition with random test cases."""
        U_arr = np.array(U_mat, dtype=float)
        V_arr = np.array(V_mat, dtype=float)
        W_arr = np.array(W_mat, dtype=float)
        for trial in range(5):
            A_test = np.random.randint(-3, 4, size=(n, m)).astype(float)
            B_test = np.random.randint(-3, 4, size=(m, p)).astype(float)
            C_expected = A_test @ B_test
            C_computed = np.zeros((n, p))
            for r in range(rank_val):
                f_r = float(np.dot(U_arr[:, r], A_test.flatten()))
                g_r = float(np.dot(V_arr[:, r], B_test.flatten()))
                if f_r == 0.0 or g_r == 0.0:
                    continue
                # W_arr[:, r] has shape (p*n,). Index k*n+i corresponds to C[i,k].
                w_col = W_arr[:, r]
                # C_computed[i,k] += w_col[k*n+i] * f_r * g_r
                C_computed += np.outer(np.ones(n), np.ones(p)) * 0  # no-op
                for i in range(n):
                    for k in range(p):
                        C_computed[i, k] += w_col[k * n + i] * f_r * g_r
            if not np.allclose(C_expected, C_computed, atol=1e-9):
                return False
        return True

    if time.time() - start_time > DEADLINE:
        return get_schoolbook()

    # Attempt Rank 65: Strassen for 4x4x4 (49 terms) + Schoolbook for 4x4x1 (16 terms)
    rank = 65
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_term(col, u_list, v_list, w_list):
        for i, j, c in u_list:
            if c != 0:
                U[i * m + j][col] += c
        for j, k, c in v_list:
            if c != 0:
                V[j * p + k][col] += c
        for k, i, c in w_list:
            if c != 0:
                W[k * n + i][col] += c

    col = 0
    
    # Strassen for 4x4 * 4x4
    # Super terms M1..M7
    # Inner terms S1..S7 for each 2x2 product
    
    # Coefficients for Strassen 2x2:
    # S1 = (P00+P11)(Q00+Q11)
    # S2 = (P10+P11)Q00
    # S3 = P00(Q01-Q11)
    # S4 = P11(Q10-Q00)
    # S5 = (P00+P01)Q11
    # S6 = (P10-P00)(Q00+Q01)
    # S7 = (P01-P11)(Q10+Q11)
    
    # Result C = [[S1+S4-S5+S7, S3+S5], [S2+S4, S1-S2+S3+S6]]
    
    # Mapping of Inner Term t to contributions to C00, C01, C10, C11 (local 2x2 result)
    inner_contribs = [
        [1, 0, 0, 1], # t=0 (S1)
        [0, 0, 1, -1], # t=1 (S2)
        [0, 1, 0, 1], # t=2 (S3)
        [1, 0, 1, 0], # t=3 (S4)
        [-1, 1, 0, 0], # t=4 (S5)
        [0, 0, 0, 1], # t=5 (S6)
        [1, 0, 0, 0]  # t=6 (S7)
    ]
    
    # Super terms M1..M7 define P and Q blocks
    super_defs = [
        # [P_blocks (A00, A01, A10, A11), Q_blocks (B00, B01, B10, B11)]
        ([1, 0, 0, 1], [1, 0, 0, 1]), # M1
        ([0, 0, 1, 1], [1, 0, 0, 0]), # M2
        ([1, 0, 0, 0], [0, 1, 0, -1]), # M3
        ([0, 0, 0, 1], [-1, 0, 1, 0]), # M4
        ([1, 1, 0, 0], [0, 0, 0, 1]), # M5
        ([-1, 0, 1, 0], [1, 1, 0, 0]), # M6
        ([0, 1, 0, -1], [0, 0, 1, 1])  # M7
    ]
    
    # Contribution of Super Term M_s to Global C blocks (C00, C01, C10, C11)
    super_contribs = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, -1], # M2
        [0, 1, 0, 1], # M3
        [1, 0, 1, 0], # M4
        [-1, 1, 0, 0], # M5
        [0, 0, 0, 1], # M6
        [1, 0, 0, 0]  # M7
    ]
    
    # Inner Strassen term definitions
    inner_defs = [
        # [P_elems (idx, coeff), Q_elems (idx, coeff)]
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)]), # S1
        ([(2, 1), (3, 1)], [(0, 1)]),          # S2
        ([(0, 1)], [(1, 1), (3, -1)]),         # S3
        ([(3, 1)], [(2, 1), (0, -1)]),         # S4
        ([(0, 1), (1, 1)], [(3, 1)]),          # S5
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)]), # S6
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)])  # S7
    ]
    
    for s_idx in range(7):
        p_blocks, q_blocks = super_defs[s_idx]
        s_contribs = super_contribs[s_idx]
        
        for t_idx in range(7):
            if time.time() - start_time > DEADLINE:
                return get_schoolbook()
                
            p_elems, q_elems = inner_defs[t_idx]
            t_contribs = inner_contribs[t_idx]
            
            u_list = []
            v_list = []
            w_list = []
            
            # Expand U
            for p_idx, p_coeff in p_elems:
                pr = p_idx // 2
                pc = p_idx % 2
                
                for b_idx, b_coeff in enumerate(p_blocks):
                    if b_coeff == 0: continue
                    base_r = (b_idx // 2) * 2
                    base_c = (b_idx % 2) * 2
                    gi = base_r + pr
                    gj = base_c + pc
                    u_list.append((gi, gj, b_coeff * p_coeff))
                    
            # Expand V
            for q_idx, q_coeff in q_elems:
                qr = q_idx // 2
                qc = q_idx % 2
                
                for b_idx, b_coeff in enumerate(q_blocks):
                    if b_coeff == 0: continue
                    base_r = (b_idx // 2) * 2
                    base_c = (b_idx % 2) * 2
                    gj = base_r + qr
                    gk = base_c + qc
                    v_list.append((gj, gk, b_coeff * q_coeff))
                    
            # Expand W
            for s_pos in range(4):
                t_c = t_contribs[s_pos]
                if t_c == 0: continue
                sr = s_pos // 2
                sc = s_pos % 2
                
                for c_block in range(4):
                    s_c = s_contribs[c_block]
                    if s_c == 0: continue
                    
                    total_coeff = t_c * s_c
                    if total_coeff == 0: continue
                    
                    base_i = (c_block // 2) * 2
                    base_k = (c_block % 2) * 2
                    gi = base_i + sr
                    gk = base_k + sc
                    
                    w_list.append((gk, gi, total_coeff))
                    
            add_term(col, u_list, v_list, w_list)
            col += 1
            
    # Part 2: Schoolbook for column 4
    for i in range(n):
        for j in range(m):
            if time.time() - start_time > DEADLINE:
                return get_schoolbook()
            u_list = [(i, j, 1)]
            v_list = [(j, 4, 1)]
            w_list = [(4, i, 1)]
            add_term(col, u_list, v_list, w_list)
            col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
