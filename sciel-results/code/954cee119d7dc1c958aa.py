"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 5
    start_time = time.time()
    deadline = start_time + 100  # Safety margin before 120s limit
    
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
    
    # Part 1: 4x4 * 4x4 using nested Strassen (rank 49)
    # Super-level: 7 Strassen terms for 2x2 blocks
    # Inner-level: 7 Strassen terms for scalar 2x2 products
    
    # Strassen for 2x2: 
    # M1=(A00+A11)(B00+B11), M2=(A10+A11)B00, M3=A00(B01-B11)
    # M4=A11(B10-B00), M5=(A00+A01)B11, M6=(A10-A00)(B00+B01)
    # M7=(A01-A11)(B10+B11)
    # C00=M1+M4-M5+M7, C01=M3+M5, C10=M2+M4, C11=M1-M2+M3+M6
    
    # Super-level block coefficients: [A00, A01, A10, A11]
    super_p_blocks = [
        [1, 0, 0, 1],   # M1: A00+A11
        [0, 0, 1, 1],   # M2: A10+A11
        [1, 0, 0, 0],   # M3: A00
        [0, 0, 0, 1],   # M4: A11
        [1, 1, 0, 0],   # M5: A00+A01
        [-1, 0, 1, 0],  # M6: A10-A00
        [0, 1, 0, -1],  # M7: A01-A11
    ]
    # Super-level block coefficients: [B00, B01, B10, B11]
    super_q_blocks = [
        [1, 0, 0, 1],   # M1: B00+B11
        [1, 0, 0, 0],   # M2: B00
        [0, 1, 0, -1],  # M3: B01-B11
        [-1, 0, 1, 0],  # M4: B10-B00
        [0, 0, 0, 1],   # M5: B11
        [1, 1, 0, 0],   # M6: B00+B01
        [0, 0, 1, 1],   # M7: B10+B11
    ]
    
    # Inner Strassen: for P=[[p00,p01],[p10,p11]], Q=[[q00,q01],[q10,q11]]
    # s1=(p00+p11)(q00+q11), s2=(p10+p11)q00, s3=p00(q01-q11)
    # s4=p11(q10-q00), s5=(p00+p01)q11, s6=(p10-p00)(q00+q01)
    # s7=(p01-p11)(q10+q11)
    # S00=s1+s4-s5+s7, S01=s3+s5, S10=s2+s4, S11=s1-s2+s3+s6
    
    # Inner P element coefficients: [p00, p01, p10, p11]
    inner_p_elems = [
        [1, 0, 0, 1],   # s1: p00+p11
        [0, 0, 1, 1],   # s2: p10+p11
        [1, 0, 0, 0],   # s3: p00
        [0, 0, 0, 1],   # s4: p11
        [1, 1, 0, 0],   # s5: p00+p01
        [-1, 0, 1, 0],  # s6: p10-p00
        [0, 1, 0, -1],  # s7: p01-p11
    ]
    # Inner Q element coefficients: [q00, q01, q10, q11]
    inner_q_elems = [
        [1, 0, 0, 1],   # s1: q00+q11
        [1, 0, 0, 0],   # s2: q00
        [0, 1, 0, -1],  # s3: q01-q11
        [-1, 0, 1, 0],  # s4: q10-q00
        [0, 0, 0, 1],   # s5: q11
        [1, 1, 0, 0],   # s6: q00+q01
        [0, 0, 1, 1],   # s7: q10+q11
    ]
    # Inner S contribution: [S00, S01, S10, S11]
    inner_s_contrib = [
        [1, 0, 0, 1],   # s1
        [0, 0, 1, -1],  # s2
        [0, 1, 0, 1],   # s3
        [1, 0, 1, 0],   # s4
        [-1, 1, 0, 0],  # s5
        [0, 0, 0, 1],   # s6
        [1, 0, 0, 0],   # s7
    ]
    
    # Super C block contribution: [C00, C01, C10, C11]
    super_c_contrib = [
        [1, 0, 0, 1],   # M1
        [0, 0, 1, -1],  # M2
        [0, 1, 0, 1],   # M3
        [1, 0, 1, 0],   # M4
        [-1, 1, 0, 0],  # M5
        [0, 0, 0, 1],   # M6
        [1, 0, 0, 0],   # M7
    ]
    
    def get_p_global(p_elem_idx, p_blocks):
        """Get global (i,j) indices for a P element."""
        res = []
        r = p_elem_idx // 2
        c = p_elem_idx % 2
        for b_idx, coeff in enumerate(p_blocks):
            if coeff == 0:
                continue
            if b_idx == 0:    # A00: i in 0,1; j in 0,1
                gi, gj = r, c
            elif b_idx == 1:  # A01: i in 0,1; j in 2,3
                gi, gj = r, 2 + c
            elif b_idx == 2:  # A10: i in 2,3; j in 0,1
                gi, gj = 2 + r, c
            elif b_idx == 3:  # A11: i in 2,3; j in 2,3
                gi, gj = 2 + r, 2 + c
            res.append((gi, gj, coeff))
        return res
    
    def get_q_global(q_elem_idx, q_blocks):
        """Get global (j,k) indices for a Q element."""
        res = []
        r = q_elem_idx // 2
        c = q_elem_idx % 2
        for b_idx, coeff in enumerate(q_blocks):
            if coeff == 0:
                continue
            if b_idx == 0:    # B00: j in 0,1; k in 0,1
                gj, gk = r, c
            elif b_idx == 1:  # B01: j in 0,1; k in 2,3
                gj, gk = r, 2 + c
            elif b_idx == 2:  # B10: j in 2,3; k in 0,1
                gj, gk = 2 + r, c
            elif b_idx == 3:  # B11: j in 2,3; k in 2,3
                gj, gk = 2 + r, 2 + c
            res.append((gj, gk, coeff))
        return res
    
    for s in range(7):
        for t in range(7):
            if time.time() > deadline:
                break
            
            p_blocks = super_p_blocks[s]
            q_blocks = super_q_blocks[s]
            
            # Build u_list from inner_p_elems[t] applied to p_blocks
            u_list = []
            for pe_idx, pe_coeff in enumerate(inner_p_elems[t]):
                if pe_coeff == 0:
                    continue
                for gi, gj, gcoeff in get_p_global(pe_idx, p_blocks):
                    u_list.append((gi, gj, pe_coeff * gcoeff))
            
            # Build v_list from inner_q_elems[t] applied to q_blocks
            v_list = []
            for qe_idx, qe_coeff in enumerate(inner_q_elems[t]):
                if qe_coeff == 0:
                    continue
                for gj, gk, gcoeff in get_q_global(qe_idx, q_blocks):
                    v_list.append((gj, gk, qe_coeff * gcoeff))
            
            # Build w_list
            w_list = []
            for s_pos in range(4):
                s_coeff = inner_s_contrib[t][s_pos]
                if s_coeff == 0:
                    continue
                s_r = s_pos // 2
                s_c = s_pos % 2
                for c_b in range(4):
                    c_coeff = super_c_contrib[s][c_b]
                    if c_coeff == 0:
                        continue
                    total_coeff = s_coeff * c_coeff
                    if total_coeff == 0:
                        continue
                    if c_b == 0:    # C00: i in 0,1; k in 0,1
                        gi, gk = s_r, s_c
                    elif c_b == 1:  # C01: i in 0,1; k in 2,3
                        gi, gk = s_r, 2 + s_c
                    elif c_b == 2:  # C10: i in 2,3; k in 0,1
                        gi, gk = 2 + s_r, s_c
                    elif c_b == 3:  # C11: i in 2,3; k in 2,3
                        gi, gk = 2 + s_r, 2 + s_c
                    w_list.append((gk, gi, total_coeff))
            
            add_term(col, u_list, v_list, w_list)
            col += 1
    
    # Part 2: 4x4 * 4x1 schoolbook (rank 16)
    for i in range(4):
        for j in range(4):
            if time.time() > deadline:
                break
            u_list = [(i, j, 1)]
            v_list = [(j, 4, 1)]
            w_list = [(4, i, 1)]
            add_term(col, u_list, v_list, w_list)
            col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
