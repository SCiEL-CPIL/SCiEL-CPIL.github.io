import numpy as np
import time

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 4, 4, 5
    rank_4x4x4 = 49
    rank_tail = 4
    total_rank = rank_4x4x4 + rank_tail
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # --- Part 1: Rank 49 decomposition for 4x4x4 ---
    
    # Define block indices for A (4x4) and B (4x4)
    # A blocks: A00, A01, A10, A11
    # A00: indices 0,1,4,5
    # A01: indices 2,3,6,7
    # A10: indices 8,9,12,13
    # A11: indices 10,11,14,15
    
    # We represent a linear form on A as an array of 16 floats.
    def zero_form_A():
        return [0.0] * 16
        
    def zero_form_B():
        return [0.0] * 20

    # Helper to get block form
    def get_block_A(b_row, b_col):
        # b_row, b_col in {0,1}
        form = [0.0] * 16
        for r in range(2):
            for c in range(2):
                i = b_row * 2 + r
                j = b_col * 2 + c
                idx = i * 4 + j
                form[idx] = 1.0
        return form

    def get_block_B(b_row, b_col):
        form = [0.0] * 20
        for r in range(2):
            for c in range(2):
                i = b_row * 2 + r
                j = b_col * 2 + c
                idx = i * 4 + j
                form[idx] = 1.0
        return form

    def add_form(form1, form2, coeff=1.0):
        res = [0.0] * len(form1)
        for i in range(len(form1)):
            res[i] = form1[i] + coeff * form2[i]
        return res

    def scale_form(form, coeff):
        return [coeff * x for x in form]

    # Define P_k and Q_k
    # P_k is 2x2 matrix of linear forms on A
    # Q_k is 2x2 matrix of linear forms on B
    
    P = []
    Q = []
    
    # k=1: P1 = A00 + A11, Q1 = B00 + B11
    A00 = get_block_A(0,0); A01 = get_block_A(0,1); A10 = get_block_A(1,0); A11 = get_block_A(1,1)
    B00 = get_block_B(0,0); B01 = get_block_B(0,1); B10 = get_block_B(1,0); B11 = get_block_B(1,1)
    
    P1 = [[add_form(A00, A11), add_form(A00, A11), add_form(A00, A11), add_form(A00, A11)]] # Wait, P_k is a matrix
    # P_k is 2x2.
    # P1 = A00 + A11. This means P1[0,0]=A00+A11, P1[0,1]=A00+A11? No.
    # Strassen: P1 = A00 + A11. This is a block matrix addition.
    # So P1[0,0] = A00[0,0] + A11[0,0].
    # A00 and A11 are 2x2 blocks.
    # So P1 is a 2x2 matrix where each entry is a sum of corresponding entries of A00 and A11.
    # In our linear form representation, A00 is a vector of 4 non-zeros.
    # P1[0,0] corresponds to the (0,0) entry of the block P1.
    # The (0,0) entry of block P1 is A00[0,0] + A11[0,0].
    # A00[0,0] is A[0,0] (index 0). A11[0,0] is A[2,2] (index 10).
    # So P1[0,0] form has 1 at 0 and 10.
    
    def get_block_entry(b_row, b_col, r, c):
        # Returns form for entry (r,c) of block (b_row, b_col)
        form = [0.0] * 16
        i = b_row * 2 + r
        j = b_col * 2 + c
        idx = i * 4 + j
        form[idx] = 1.0
        return form

    def get_block_entry_B(b_row, b_col, r, c):
        form = [0.0] * 20
        i = b_row * 2 + r
        j = b_col * 2 + c
        idx = i * 4 + j
        form[idx] = 1.0
        return form

    # Build P and Q matrices of forms
    # P[k][r][c] is the form for entry (r,c) of block P_k
    P = [[None for _ in range(2)] for _ in range(7)]
    Q = [[None for _ in range(2)] for _ in range(7)]
    
    for k in range(7):
        for r in range(2):
            for c in range(2):
                P[k][r][c] = [0.0] * 16
                Q[k][r][c] = [0.0] * 20

    # k=0 (1st)
    # P0 = A00 + A11
    for r in range(2):
        for c in range(2):
            P[0][r][c] = add_form(get_block_entry(0,0,r,c), get_block_entry(1,1,r,c))
            Q[0][r][c] = add_form(get_block_entry_B(0,0,r,c), get_block_entry_B(1,1,r,c))
            
    # k=1 (2nd)
    # P1 = A10 + A11
    # Q1 = B00
    for r in range(2):
        for c in range(2):
            P[1][r][c] = add_form(get_block_entry(1,0,r,c), get_block_entry(1,1,r,c))
            Q[1][r][c] = get_block_entry_B(0,0,r,c)
            
    # k=2 (3rd)
    # P2 = A00
    # Q2 = B01 - B11
    for r in range(2):
        for c in range(2):
            P[2][r][c] = get_block_entry(0,0,r,c)
            Q[2][r][c] = add_form(get_block_entry_B(0,1,r,c), get_block_entry_B(1,1,r,c), -1.0)
            
    # k=3 (4th)
    # P3 = A11
    # Q3 = B10 - B00
    for r in range(2):
        for c in range(2):
            P[3][r][c] = get_block_entry(1,1,r,c)
            Q[3][r][c] = add_form(get_block_entry_B(1,0,r,c), get_block_entry_B(0,0,r,c), -1.0)
            
    # k=4 (5th)
    # P4 = A00 + A01
    # Q4 = B11
    for r in range(2):
        for c in range(2):
            P[4][r][c] = add_form(get_block_entry(0,0,r,c), get_block_entry(0,1,r,c))
            Q[4][r][c] = get_block_entry_B(1,1,r,c)
            
    # k=5 (6th)
    # P5 = A10 - A00
    # Q5 = B00 + B01
    for r in range(2):
        for c in range(2):
            P[5][r][c] = add_form(get_block_entry(1,0,r,c), get_block_entry(0,0,r,c), -1.0)
            Q[5][r][c] = add_form(get_block_entry_B(0,0,r,c), get_block_entry_B(0,1,r,c))
            
    # k=6 (7th)
    # P6 = A01 - A11
    # Q6 = B10 + B11
    for r in range(2):
        for c in range(2):
            P[6][r][c] = add_form(get_block_entry(0,1,r,c), get_block_entry(1,1,r,c), -1.0)
            Q[6][r][c] = add_form(get_block_entry_B(1,0,r,c), get_block_entry_B(1,1,r,c))

    # Inner Strassen coefficients
    # m1 = (x0+x3)(y0+y3)
    # m2 = (x2+x3)y0
    # m3 = x0(y1-y3)
    # m4 = x3(y2-y0)
    # m5 = (x0+x1)y3
    # m6 = (x2-x0)(y0+y1)
    # m7 = (x1-x3)(y2+y3)
    
    # x0=P[r][c] for (0,0), x1=(0,1), x2=(1,0), x3=(1,1)
    
    inner_mults = [
        ((0,0), (1,1), 1.0, (0,0), (1,1), 1.0), # m1
        ((1,0), (1,1), 1.0, (0,0), (0,0), 1.0), # m2 - wait, (x2+x3)y0. x2=(1,0), x3=(1,1). y0=(0,0).
        ((0,0), (0,0), 1.0, (0,1), (1,1), -1.0), # m3 - x0(y1-y3).
        ((1,1), (1,1), 1.0, (1,0), (0,0), -1.0), # m4 - x3(y2-y0).
        ((0,0), (0,1), 1.0, (1,1), (1,1), 1.0), # m5 - (x0+x1)y3.
        ((1,0), (0,0), -1.0, (0,0), (0,1), 1.0), # m6 - (x2-x0)(y0+y1).
        ((0,1), (1,1), -1.0, (1,0), (1,1), 1.0), # m7 - (x1-x3)(y2+y3).
    ]
    # Note: the tuple is (x_row, x_col, x_coeff, y_row, y_col, y_coeff) is not enough for sums.
    # Let's just construct the forms explicitly.

    def get_inner_forms(k, l):
        x0 = P[k][0][0]; x1 = P[k][0][1]; x2 = P[k][1][0]; x3 = P[k][1][1]
        y0 = Q[k][0][0]; y1 = Q[k][0][1]; y2 = Q[k][1][0]; y3 = Q[k][1][1]
        
        if l == 0: # m1
            la = add_form(add_form(x0, x3), zero_form_A())
            lb = add_form(add_form(y0, y3), zero_form_B())
        elif l == 1: # m2
            la = add_form(x2, x3)
            lb = y0
        elif l == 2: # m3
            la = x0
            lb = add_form(y1, y3, -1.0)
        elif l == 3: # m4
            la = x3
            lb = add_form(y2, y0, -1.0)
        elif l == 4: # m5
            la = add_form(x0, x1)
            lb = y3
        elif l == 5: # m6
            la = add_form(x2, x0, -1.0)
            lb = add_form(y0, y1)
        elif l == 6: # m7
            la = add_form(x1, x3, -1.0)
            lb = add_form(y2, y3)
        return la, lb

    # Output contributions
    # z0 = m1 + m4 - m5 + m7
    # z1 = m3 + m5
    # z2 = m2 + m4
    # z3 = m1 - m2 + m3 + m6
    
    # m_l contributes to z_j with coeff c
    m_to_z = {
        0: [(0, 1.0), (3, 1.0)], # m1
        1: [(2, 1.0), (3, -1.0)], # m2
        2: [(1, 1.0), (3, 1.0)], # m3
        3: [(0, 1.0), (2, 1.0)], # m4
        4: [(0, -1.0), (1, 1.0)], # m5
        5: [(3, 1.0)], # m6
        6: [(0, 1.0)] # m7
    }
    
    # M_k contributes to C blocks
    # M0 (1st): C00(+1), C11(+1)
    # M1 (2nd): C10(+1), C11(-1)
    # M2 (3rd): C01(+1), C11(+1)
    # M3 (4th): C00(+1), C10(+1)
    # M4 (5th): C00(-1), C01(+1)
    # M5 (6th): C11(+1)
    # M6 (7th): C00(+1)
    
    m_to_c_blocks = {
        0: [(0,0, 1.0), (1,1, 1.0)],
        1: [(1,0, 1.0), (1,1, -1.0)],
        2: [(0,1, 1.0), (1,1, 1.0)],
        3: [(0,0, 1.0), (1,0, 1.0)],
        4: [(0,0, -1.0), (0,1, 1.0)],
        5: [(1,1, 1.0)],
        6: [(0,0, 1.0)]
    }
    
    # Block (br, bc) entry (r, c) maps to C[i, k]
    # i = br*2 + r, k = bc*2 + c
    # W index = k * 4 + i
    
    for k in range(7):
        for l in range(7):
            r_idx = k * 7 + l
            la, lb = get_inner_forms(k, l)
            
            # Fill U
            for i in range(16):
                if la[i] != 0:
                    U[i][r_idx] = la[i]
                    
            # Fill V
            for i in range(20):
                if lb[i] != 0:
                    V[i][r_idx] = lb[i]
            
            # Fill W
            # la, lb correspond to m_l
            # m_l contributes to z_j
            for z_j, c_coeff in m_to_z[l]:
                # z_j is entry (r,c) of M_k
                zr = z_j // 2
                zc = z_j % 2
                # M_k contributes to C blocks
                for c_br, c_bc, d_coeff in m_to_c_blocks[k]:
                    # Entry in C is at (c_br*2+zr, c_bc*2+zc)
                    c_i = c_br * 2 + zr
                    c_k = c_bc * 2 + zc
                    w_idx = c_k * 4 + c_i
                    w_val = c_coeff * d_coeff
                    if w_val != 0:
                        W[w_idx][r_idx] = w_val

    # --- Part 2: Rank 4 decomposition for 5th column ---
    for i in range(4):
        r_idx = 49 + i
        for j in range(4):
            U[i * 4 + j][r_idx] = 1
            V[j * 5 + 4][r_idx] = 1
        W[4 * 4 + i][r_idx] = 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
