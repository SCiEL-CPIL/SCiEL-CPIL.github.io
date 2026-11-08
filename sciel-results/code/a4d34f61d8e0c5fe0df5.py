"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    _t0 = time.time()
    DEADLINE = 120.0  # seconds; evaluator kills at 150s, keep margin

    n, m, p = DIMENSIONS  # 4, 4, 5
    
    # Rank 65 decomposition:
    # 49 terms for 4x4 * 4x4 (Strassen)
    # 16 terms for 4x4 * 4x1 (Schoolbook)
    
    rank = 65
    U = [[0] * rank for _ in range(n * m)]  # 16 x 65
    V = [[0] * rank for _ in range(m * p)]  # 20 x 65
    W = [[0] * rank for _ in range(p * n)]  # 20 x 65
    
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
    
    # Strassen decomposition for 4x4 * 4x4
    # Split into 2x2 blocks and apply Strassen at two levels
    
    # Super-level Strassen: 7 terms, each involves 2x2 matrix products
    # p_blocks_table[s] = coefficients for A blocks [A00, A01, A10, A11]
    # q_blocks_table[s] = coefficients for B blocks [B00, B01, B10, B11]
    # c_block_contrib[s] = how super-term s contributes to C blocks [C00, C01, C10, C11]
    
    p_blocks_table = [
        [1, 0, 0, 1],   # S0: A00 + A11
        [0, 0, 1, 1],   # S1: A10 + A11
        [1, 0, 0, 0],   # S2: A00
        [0, 0, 0, 1],   # S3: A11
        [1, 1, 0, 0],   # S4: A00 + A01
        [-1, 0, 1, 0],  # S5: A10 - A00
        [0, 1, 0, -1],  # S6: A01 - A11
    ]
    
    q_blocks_table = [
        [1, 0, 0, 1],   # S0: B00 + B11
        [1, 0, 0, 0],   # S1: B00
        [0, 1, 0, -1],  # S2: B01 - B11
        [-1, 0, 1, 0],  # S3: B10 - B00
        [0, 0, 0, 1],   # S4: B11
        [1, 1, 0, 0],   # S5: B00 + B01
        [0, 0, 1, 1],   # S6: B10 + B11
    ]
    
    c_block_contrib_table = [
        [1, 0, 0, 1],   # S0: C00 + C11
        [0, 0, 1, -1],  # S1: C10 - C11
        [0, 1, 0, 1],   # S2: C01 + C11
        [1, 0, 1, 0],   # S3: C00 + C10
        [-1, 1, 0, 0],  # S4: -C00 + C01
        [0, 0, 0, 1],   # S5: C11
        [1, 0, 0, 0],   # S6: C00
    ]
    
    # Inner Strassen terms for 2x2 matrix products
    # Each term: (p_elems, q_elems, s_contrib)
    # p_elems: list of (elem_idx, coeff) where elem_idx: 0=P00, 1=P01, 2=P10, 3=P11
    # q_elems: list of (elem_idx, coeff)
    # s_contrib: [S00, S01, S10, S11] contribution to 2x2 result
    
    inner_table = [
        # M1 = (P00+P11)(Q00+Q11)
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [1, 0, 0, 1]),
        # M2 = (P10+P11)Q00
        ([(2, 1), (3, 1)], [(0, 1)], [0, 0, 1, -1]),
        # M3 = P00(Q01-Q11)
        ([(0, 1)], [(1, 1), (3, -1)], [0, 1, 0, 1]),
        # M4 = P11(Q10-Q00)
        ([(3, 1)], [(2, 1), (0, -1)], [1, 0, 1, 0]),
        # M5 = (P00+P01)Q11
        ([(0, 1), (1, 1)], [(3, 1)], [-1, 1, 0, 0]),
        # M6 = (P10-P00)(Q00+Q01)
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [0, 0, 0, 1]),
        # M7 = (P01-P11)(Q10+Q11)
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [1, 0, 0, 0]),
    ]
    
    # Build all 49 terms
    for s in range(7):
        if time.time() - _t0 > DEADLINE - 30:
            break
        p_blocks = p_blocks_table[s]
        q_blocks = q_blocks_table[s]
        c_block_contrib = c_block_contrib_table[s]
        
        for t in range(7):
            if time.time() - _t0 > DEADLINE - 30:
                break
            p_elems, q_elems, s_contrib = inner_table[t]
            
            u_list = []
            v_list = []
            w_list = []
            
            # Build u_list: for each P element, expand across A blocks
            for p_idx, p_coeff in p_elems:
                r = p_idx // 2
                c = p_idx % 2
                for b_idx, coeff in enumerate(p_blocks):
                    if coeff == 0:
                        continue
                    if b_idx == 0:
                        gi, gj = r, c
                    elif b_idx == 1:
                        gi, gj = r, 2 + c
                    elif b_idx == 2:
                        gi, gj = 2 + r, c
                    elif b_idx == 3:
                        gi, gj = 2 + r, 2 + c
                    u_list.append((gi, gj, coeff * p_coeff))
            
            # Build v_list: for each Q element, expand across B blocks
            for q_idx, q_coeff in q_elems:
                r = q_idx // 2
                c = q_idx % 2
                for b_idx, coeff in enumerate(q_blocks):
                    if coeff == 0:
                        continue
                    if b_idx == 0:
                        gj, gk = r, c
                    elif b_idx == 1:
                        gj, gk = r, 2 + c
                    elif b_idx == 2:
                        gj, gk = 2 + r, c
                    elif b_idx == 3:
                        gj, gk = 2 + r, 2 + c
                    v_list.append((gj, gk, coeff * q_coeff))
            
            # Build w_list: for each S element, expand across C blocks
            for s_pos in range(4):
                s_coeff = s_contrib[s_pos]
                if s_coeff == 0:
                    continue
                s_r = s_pos // 2
                s_c = s_pos % 2
                
                for c_b in range(4):
                    c_coeff = c_block_contrib[c_b]
                    if c_coeff == 0:
                        continue
                    
                    total_coeff = s_coeff * c_coeff
                    if total_coeff == 0:
                        continue
                    
                    if c_b == 0:
                        gi, gk = s_r, s_c
                    elif c_b == 1:
                        gi, gk = s_r, 2 + s_c
                    elif c_b == 2:
                        gi, gk = 2 + s_r, s_c
                    elif c_b == 3:
                        gi, gk = 2 + s_r, 2 + s_c
                    
                    w_list.append((gk, gi, total_coeff))
            
            add_term(col, u_list, v_list, w_list)
            col += 1
    
    # Part 2: 4x4 * 4x1 using Schoolbook for the 5th column
    # C[i,4] = sum_j A[i,j] * B[j,4]
    for i in range(4):
        for j in range(4):
            if time.time() - _t0 > DEADLINE - 10:
                break
            u_list = [(i, j, 1)]
            v_list = [(j, 4, 1)]
            w_list = [(4, i, 1)]
            add_term(col, u_list, v_list, w_list)
            col += 1
        if time.time() - _t0 > DEADLINE - 10:
            break
    
    # Verify decomposition correctness
    if time.time() - _t0 < DEADLINE - 5:
        try:
            U_np = np.array(U, dtype=float)
            V_np = np.array(V, dtype=float)
            W_np = np.array(W, dtype=float)
            
            rng = np.random.RandomState(42)
            A_test = rng.randint(-3, 4, size=(n, m)).astype(float)
            B_test = rng.randint(-3, 4, size=(m, p)).astype(float)
            C_expected = A_test @ B_test
            
            C_computed = np.zeros((n, p), dtype=float)
            for r in range(rank):
                u_mat = U_np[:, r].reshape(n, m)
                v_mat = V_np[:, r].reshape(m, p)
                w_mat = W_np[:, r].reshape(p, n).T
                C_computed += u_mat @ v_mat * w_mat
            
            if not np.allclose(C_computed, C_expected, atol=1e-9):
                # If verification fails, fall back to pure schoolbook
                U_fb = [[0] * (n*m*p) for _ in range(n*m)]
                V_fb = [[0] * (n*m*p) for _ in range(m*p)]
                W_fb = [[0] * (n*m*p) for _ in range(p*n)]
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            idx = i*m*p + j*p + k
                            U_fb[i*m+j][idx] = 1
                            V_fb[j*p+k][idx] = 1
                            W_fb[k*n+i][idx] = 1
                return {"U": U_fb, "V": V_fb, "W": W_fb}
        except Exception:
            pass
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
