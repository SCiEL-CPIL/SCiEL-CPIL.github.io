"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add a linear form (list of 16) to a target vector
    def add_vec(target, source, coeff=1):
        if coeff == 0:
            return
        for i in range(16):
            target[i] += coeff * source[i]

    # Define identity linear forms for A and B
    # A indices: 0..15 (row-major: i*4+j)
    # B indices: 0..15 (row-major: j*4+k)
    # We need linear forms for the entries of the 2x2 blocks.
    
    # Block A_00: A[0,0], A[0,1], A[1,0], A[1,1] -> indices 0, 1, 4, 5
    # Block A_01: A[0,2], A[0,3], A[1,2], A[1,3] -> indices 2, 3, 6, 7
    # Block A_10: A[2,0], A[2,1], A[3,0], A[3,1] -> indices 8, 9, 12, 13
    # Block A_11: A[2,2], A[2,3], A[3,2], A[3,3] -> indices 10, 11, 14, 15
    
    # Let's define the 4 linear forms for each block entry (00, 01, 10, 11)
    # We'll store them as lists of 16 floats.
    
    def get_A_block_form(block_r, block_c, entry_r, entry_c):
        # block_r, block_c in {0,1}
        # entry_r, entry_c in {0,1}
        # Global row = block_r * 2 + entry_r
        # Global col = block_c * 2 + entry_c
        # Index = row * 4 + col
        idx = (block_r * 2 + entry_r) * 4 + (block_c * 2 + entry_c)
        vec = [0] * 16
        vec[idx] = 1
        return vec

    def get_B_block_form(block_r, block_c, entry_r, entry_c):
        # B is m x p = 4x4.
        # Block r, c. Entry r, c.
        # Global row (of B) = block_r * 2 + entry_r (this is the 'j' index)
        # Global col (of B) = block_c * 2 + entry_c (this is the 'k' index)
        # Index = row * 4 + col
        idx = (block_r * 2 + entry_r) * 4 + (block_c * 2 + entry_c)
        vec = [0] * 16
        vec[idx] = 1
        return vec

    # Construct Q_s and R_s matrices (2x2 of linear forms)
    # Q_s is a list of 4 vectors: [00, 01, 10, 11]
    
    # M1: (A00 + A11) (B00 + B11)
    Q1 = []
    R1 = []
    for er in range(2):
        for ec in range(2):
            qa = get_A_block_form(0, 0, er, ec)
            qb = get_A_block_form(1, 1, er, ec)
            q = [0]*16
            add_vec(q, qa, 1)
            add_vec(q, qb, 1)
            Q1.append(q)
            
            ra = get_B_block_form(0, 0, er, ec)
            rb = get_B_block_form(1, 1, er, ec)
            r = [0]*16
            add_vec(r, ra, 1)
            add_vec(r, rb, 1)
            R1.append(r)

    # M2: (A10 + A11) B00
    Q2 = []
    R2 = []
    for er in range(2):
        for ec in range(2):
            qa = get_A_block_form(1, 0, er, ec)
            qb = get_A_block_form(1, 1, er, ec)
            q = [0]*16
            add_vec(q, qa, 1)
            add_vec(q, qb, 1)
            Q2.append(q)
            
            ra = get_B_block_form(0, 0, er, ec)
            R2.append(ra)

    # M3: A00 (B01 - B11)
    Q3 = []
    R3 = []
    for er in range(2):
        for ec in range(2):
            Q3.append(get_A_block_form(0, 0, er, ec))
            
            ra = get_B_block_form(0, 1, er, ec)
            rb = get_B_block_form(1, 1, er, ec)
            r = [0]*16
            add_vec(r, ra, 1)
            add_vec(r, rb, -1)
            R3.append(r)

    # M4: A11 (B10 - B00)
    Q4 = []
    R4 = []
    for er in range(2):
        for ec in range(2):
            Q4.append(get_A_block_form(1, 1, er, ec))
            
            ra = get_B_block_form(1, 0, er, ec)
            rb = get_B_block_form(0, 0, er, ec)
            r = [0]*16
            add_vec(r, ra, 1)
            add_vec(r, rb, -1)
            R4.append(r)

    # M5: (A00 + A01) B11
    Q5 = []
    R5 = []
    for er in range(2):
        for ec in range(2):
            qa = get_A_block_form(0, 0, er, ec)
            qb = get_A_block_form(0, 1, er, ec)
            q = [0]*16
            add_vec(q, qa, 1)
            add_vec(q, qb, 1)
            Q5.append(q)
            
            R5.append(get_B_block_form(1, 1, er, ec))

    # M6: (A10 - A00) (B00 + B01)
    Q6 = []
    R6 = []
    for er in range(2):
        for ec in range(2):
            qa = get_A_block_form(1, 0, er, ec)
            qb = get_A_block_form(0, 0, er, ec)
            q = [0]*16
            add_vec(q, qa, 1)
            add_vec(q, qb, -1)
            Q6.append(q)
            
            ra = get_B_block_form(0, 0, er, ec)
            rb = get_B_block_form(0, 1, er, ec)
            r = [0]*16
            add_vec(r, ra, 1)
            add_vec(r, rb, 1)
            R6.append(r)

    # M7: (A01 - A11) (B10 + B11)
    Q7 = []
    R7 = []
    for er in range(2):
        for ec in range(2):
            qa = get_A_block_form(0, 1, er, ec)
            qb = get_A_block_form(1, 1, er, ec)
            q = [0]*16
            add_vec(q, qa, 1)
            add_vec(q, qb, -1)
            Q7.append(q)
            
            ra = get_B_block_form(1, 0, er, ec)
            rb = get_B_block_form(1, 1, er, ec)
            r = [0]*16
            add_vec(r, ra, 1)
            add_vec(r, rb, 1)
            R7.append(r)

    Qs = [Q1, Q2, Q3, Q4, Q5, Q6, Q7]
    Rs = [R1, R2, R3, R4, R5, R6, R7]

    # Strassen reconstruction coefficients for M_s entries
    # M_00 = m1 + m4 - m5 + m7
    # M_01 = m3 + m5
    # M_10 = m2 + m4
    # M_11 = m1 - m2 + m3 + m6
    
    # Map m_t (1-indexed 1..7) to contributions to M entries (00, 01, 10, 11)
    # m1: 00(+1), 11(+1)
    # m2: 10(+1), 11(-1)
    # m3: 01(+1), 11(+1)
    # m4: 00(+1), 10(+1)
    # m5: 00(-1), 01(+1)
    # m6: 11(+1)
    # m7: 00(+1)
    
    m_contribs = {
        1: [(0, 1), (3, 1)],   # (entry_idx, sign)
        2: [(2, 1), (3, -1)],
        3: [(1, 1), (3, 1)],
        4: [(0, 1), (2, 1)],
        5: [(0, -1), (1, 1)],
        6: [(3, 1)],
        7: [(0, 1)]
    }

    # Map M_s block to global C blocks
    # M1: C00(+1), C11(+1)
    # M2: C10(+1), C11(-1)
    # M3: C01(+1), C11(+1)
    # M4: C00(+1), C10(+1)
    # M5: C00(-1), C01(+1)
    # M6: C11(+1)
    # M7: C00(+1)
    
    # C block indices: 00=(0,0), 01=(0,1), 10=(1,0), 11=(1,1)
    # Global C index for entry (er, ec) in block (br, bc) is:
    # Row I = 2*br + er
    # Col K = 2*bc + ec
    # W index = K * 4 + I
    
    block_map = {
        1: [(0, 0, 1), (1, 1, 1)], # (br, bc, sign)
        2: [(1, 0, 1), (1, 1, -1)],
        3: [(0, 1, 1), (1, 1, 1)],
        4: [(0, 0, 1), (1, 0, 1)],
        5: [(0, 0, -1), (0, 1, 1)],
        6: [(1, 1, 1)],
        7: [(0, 0, 1)]
    }

    col = 0
    for s_idx in range(7):
        Q = Qs[s_idx]
        R = Rs[s_idx]
        # Q, R are lists of 4 vectors (00, 01, 10, 11)
        
        for t_idx in range(7):
            # Determine U and V linear forms for m_{s, t_idx+1}
            # Strassen formulas for m_t in terms of Q, R entries
            # m1 = (q00 + q11)(r00 + r11)
            # m2 = (q10 + q11)r00
            # m3 = q00(r01 - r11)
            # m4 = q11(r10 - r00)
            # m5 = (q00 + q01)r11
            # m6 = (q10 - q00)(r00 + r01)
            # m7 = (q01 - q11)(r10 + r11)
            
            u = [0]*16
            v = [0]*16
            
            if t_idx == 0: # m1
                add_vec(u, Q[0], 1); add_vec(u, Q[3], 1)
                add_vec(v, R[0], 1); add_vec(v, R[3], 1)
            elif t_idx == 1: # m2
                add_vec(u, Q[2], 1); add_vec(u, Q[3], 1)
                add_vec(v, R[0], 1)
            elif t_idx == 2: # m3
                add_vec(u, Q[0], 1)
                add_vec(v, R[1], 1); add_vec(v, R[3], -1)
            elif t_idx == 3: # m4
                add_vec(u, Q[3], 1)
                add_vec(v, R[2], 1); add_vec(v, R[0], -1)
            elif t_idx == 4: # m5
                add_vec(u, Q[0], 1); add_vec(u, Q[1], 1)
                add_vec(v, R[3], 1)
            elif t_idx == 5: # m6
                add_vec(u, Q[2], 1); add_vec(u, Q[0], -1)
                add_vec(v, R[0], 1); add_vec(v, R[1], 1)
            elif t_idx == 6: # m7
                add_vec(u, Q[1], 1); add_vec(u, Q[3], -1)
                add_vec(v, R[2], 1); add_vec(v, R[3], 1)
            
            # Store U and V
            for i in range(16):
                U[i][col] = u[i]
                V[i][col] = v[i]
                
            # Determine W contributions
            w = [0]*16
            t_num = t_idx + 1
            for entry_idx, sign_m in m_contribs[t_num]:
                # entry_idx: 0=00, 1=01, 2=10, 3=11
                er = entry_idx // 2
                ec = entry_idx % 2
                
                for br, bc, sign_b in block_map[s_idx + 1]:
                    I = 2 * br + er
                    K = 2 * bc + ec
                    w_idx = K * 4 + I
                    w[w_idx] += sign_m * sign_b
            
            for i in range(16):
                W[i][col] = w[i]
                
            col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
