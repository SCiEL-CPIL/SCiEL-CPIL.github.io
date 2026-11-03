import numpy as np

DIMENSIONS = [3, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    # Total rank 88
    rank = 88
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col_idx = 0
    
    # Helper to add a rank-1 term
    def add_term(u, v, w):
        nonlocal col_idx
        for i in range(len(u)):
            if u[i] != 0:
                U[i][col_idx] = u[i]
        for i in range(len(v)):
            if v[i] != 0:
                V[i][col_idx] = v[i]
        for i in range(len(w)):
            if w[i] != 0:
                W[i][col_idx] = w[i]
        col_idx += 1

    # Strassen decomposition for 2x2x2
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # m1 = (a+d)(e+h)
    # m2 = (c+d)e
    # m3 = a(f-h)
    # m4 = d(g-e)
    # m5 = (a+b)h
    # m6 = (c-a)(e+f)
    # m7 = (b-d)(g+h)
    # C = [[m1+m4-m5+m7, m3+m5], [m2+m4, m1-m2+m3+m6]]
    
    def add_strassen(i_a_start, j_a_start, j_b_start, k_c_start):
        # i_a_start: row index in A for the 2x2 block
        # j_a_start: col index in A for the 2x2 block
        # j_b_start: row index in B for the 2x2 block
        # k_c_start: col index in C for the 2x2 block
        
        # A indices
        a_00 = i_a_start * m + j_a_start
        a_01 = i_a_start * m + (j_a_start + 1)
        a_10 = (i_a_start + 1) * m + j_a_start
        a_11 = (i_a_start + 1) * m + (j_a_start + 1)
        
        # B indices
        b_00 = j_b_start * p + k_c_start
        b_01 = j_b_start * p + (k_c_start + 1)
        b_10 = (j_b_start + 1) * p + k_c_start
        b_11 = (j_b_start + 1) * p + (k_c_start + 1)
        
        # C indices (W index: k*n + i)
        # C is 2x2 block at rows i_a_start, i_a_start+1 and cols k_c_start, k_c_start+1
        c_00 = k_c_start * n + i_a_start
        c_01 = (k_c_start + 1) * n + i_a_start
        c_10 = k_c_start * n + (i_a_start + 1)
        c_11 = (k_c_start + 1) * n + (i_a_start + 1)
        
        # m1 = (a+d)(e+h)
        u1 = [0]*(n*m); v1 = [0]*(m*p); w1 = [0]*(p*n)
        u1[a_00] = 1; u1[a_11] = 1
        v1[b_00] = 1; v1[b_11] = 1
        w1[c_00] = 1; w1[c_11] = 1
        add_term(u1, v1, w1)
        
        # m2 = (c+d)e
        u2 = [0]*(n*m); v2 = [0]*(m*p); w2 = [0]*(p*n)
        u2[a_10] = 1; u2[a_11] = 1
        v2[b_00] = 1
        w2[c_10] = 1; w2[c_11] = -1
        add_term(u2, v2, w2)
        
        # m3 = a(f-h)
        u3 = [0]*(n*m); v3 = [0]*(m*p); w3 = [0]*(p*n)
        u3[a_00] = 1
        v3[b_01] = 1; v3[b_11] = -1
        w3[c_01] = 1; w3[c_11] = 1
        add_term(u3, v3, w3)
        
        # m4 = d(g-e)
        u4 = [0]*(n*m); v4 = [0]*(m*p); w4 = [0]*(p*n)
        u4[a_11] = 1
        v4[b_10] = 1; v4[b_00] = -1
        w4[c_00] = 1; w4[c_10] = 1
        add_term(u4, v4, w4)
        
        # m5 = (a+b)h
        u5 = [0]*(n*m); v5 = [0]*(m*p); w5 = [0]*(p*n)
        u5[a_00] = 1; u5[a_01] = 1
        v5[b_11] = 1
        w5[c_00] = -1; w5[c_01] = 1
        add_term(u5, v5, w5)
        
        # m6 = (c-a)(e+f)
        u6 = [0]*(n*m); v6 = [0]*(m*p); w6 = [0]*(p*n)
        u6[a_10] = 1; u6[a_00] = -1
        v6[b_00] = 1; v6[b_01] = 1
        w6[c_11] = 1
        add_term(u6, v6, w6)
        
        # m7 = (b-d)(g+h)
        u7 = [0]*(n*m); v7 = [0]*(m*p); w7 = [0]*(p*n)
        u7[a_01] = 1; u7[a_11] = -1
        v7[b_10] = 1; v7[b_11] = 1
        w7[c_00] = 1
        add_term(u7, v7, w7)

    # Schoolbook decomposition for 1x2x2
    # X = [x1, x2], Y = [[y1, y2], [y3, y4]], Z = [z1, z2]
    # z1 = x1 y1 + x2 y3
    # z2 = x1 y2 + x2 y4
    
    def add_schoolbook_1x2x2(i_a_start, j_a_start, j_b_start, k_c_start):
        # i_a_start: row index in A (only 1 row)
        # j_a_start: col index in A (2 cols)
        # j_b_start: row index in B (2 rows)
        # k_c_start: col index in C (2 cols)
        
        # A indices (1x2)
        a_0 = i_a_start * m + j_a_start
        a_1 = i_a_start * m + (j_a_start + 1)
        
        # B indices (2x2)
        b_00 = j_b_start * p + k_c_start
        b_01 = j_b_start * p + (k_c_start + 1)
        b_10 = (j_b_start + 1) * p + k_c_start
        b_11 = (j_b_start + 1) * p + (k_c_start + 1)
        
        # C indices (1x2)
        c_0 = k_c_start * n + i_a_start
        c_1 = (k_c_start + 1) * n + i_a_start
        
        # Term 1: x1 y1 -> z1
        u1 = [0]*(n*m); v1 = [0]*(m*p); w1 = [0]*(p*n)
        u1[a_0] = 1
        v1[b_00] = 1
        w1[c_0] = 1
        add_term(u1, v1, w1)
        
        # Term 2: x2 y3 -> z1
        u2 = [0]*(n*m); v2 = [0]*(m*p); w2 = [0]*(p*n)
        u2[a_1] = 1
        v2[b_10] = 1
        w2[c_0] = 1
        add_term(u2, v2, w2)
        
        # Term 3: x1 y2 -> z2
        u3 = [0]*(n*m); v3 = [0]*(m*p); w3 = [0]*(p*n)
        u3[a_0] = 1
        v3[b_01] = 1
        w3[c_1] = 1
        add_term(u3, v3, w3)
        
        # Term 4: x2 y4 -> z2
        u4 = [0]*(n*m); v4 = [0]*(m*p); w4 = [0]*(p*n)
        u4[a_1] = 1
        v4[b_11] = 1
        w4[c_1] = 1
        add_term(u4, v4, w4)

    # Main construction
    # A_L (cols 0,1) * B_T (rows 0,1)
    # A_L rows 0,1 (i=0,1) and row 2 (i=2)
    # B_T cols 0-1, 2-3, 4-5, 6-7
    
    for k_block in range(4):
        k_start = k_block * 2
        # A_L1 (rows 0,1) * B_T block
        add_strassen(0, 0, 0, k_start)
        # A_L2 (row 2) * B_T block
        add_schoolbook_1x2x2(2, 0, 0, k_start)
        
    # A_R (cols 2,3) * B_B (rows 2,3)
    for k_block in range(4):
        k_start = k_block * 2
        # A_R1 (rows 0,1) * B_B block
        add_strassen(0, 2, 2, k_start)
        # A_R2 (row 2) * B_B block
        add_schoolbook_1x2x2(2, 2, 2, k_start)

    return {"U": U, "V": V, "W": W}
