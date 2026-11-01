import numpy as np

DIMENSIONS = [2, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    rank = 54
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a term
    def add_term(u_indices, u_vals, v_indices, v_vals, w_indices, w_vals):
        nonlocal col
        for idx, val in zip(u_indices, u_vals):
            U[idx][col] = val
        for idx, val in zip(v_indices, v_vals):
            V[idx][col] = val
        for idx, val in zip(w_indices, w_vals):
            W[idx][col] = val
        col += 1

    # Strassen for 2x2x2
    # A indices: 00, 01, 10, 11
    # B indices: 00, 01, 10, 11
    # C indices: 00, 01, 10, 11
    def add_strassen(a_base, b_base, c_base):
        # a_base is index of A_00
        # b_base is index of B_00
        # c_base is index of C_00
        # A_00 = a_base, A_01 = a_base+1, A_10 = a_base+2, A_11 = a_base+3
        # B_00 = b_base, B_01 = b_base+1, B_10 = b_base+2, B_11 = b_base+3
        # C_00 = c_base, C_01 = c_base+1, C_10 = c_base+2, C_11 = c_base+3
        
        # M1 = (A00 + A11)(B00 + B11)
        add_term([a_base, a_base+3], [1, 1], [b_base, b_base+3], [1, 1], [c_base, c_base+3], [1, 1])
        # M2 = (A10 + A11)B00
        add_term([a_base+2, a_base+3], [1, 1], [b_base], [1], [c_base+2], [1])
        # M3 = (A00 - A01)(B10 + B11)
        add_term([a_base, a_base+1], [1, -1], [b_base+2, b_base+3], [1, 1], [c_base+1], [1])
        # M4 = (A11 - A01)(B01 + B11)
        add_term([a_base+3, a_base+1], [1, -1], [b_base+1, b_base+3], [1, 1], [c_base], [1])
        # M5 = (A00 + A10)B11
        add_term([a_base, a_base+2], [1, 1], [b_base+3], [1], [c_base], [-1])
        # M6 = (A01 + A10)(B00 - B10)
        add_term([a_base+1, a_base+2], [1, 1], [b_base, b_base+2], [1, -1], [c_base+3], [1])
        # M7 = (A00 - A11)B01
        add_term([a_base, a_base+3], [1, -1], [b_base+1], [1], [c_base+3], [1])

    # Block 1: A_1 (2x2) cols 0,1. B_1 (2x6) rows 0,1.
    # A_1 indices: 0,1,2,3
    # B_1 indices: 0..5 (row 0), 6..11 (row 1)
    # Split B_1 into 3 2x2 blocks.
    # B_11: cols 0,1 of B_1. Indices 0,1,6,7
    # B_12: cols 2,3 of B_1. Indices 2,3,8,9
    # B_13: cols 4,5 of B_1. Indices 4,5,10,11
    # C_1: cols 0,1,2,3,4,5 of C.
    # C is 2x6.
    # C_11: C cols 0,1. Indices 0,1 (col 0), 2,3 (col 1)
    # C_12: C cols 2,3. Indices 4,5 (col 2), 6,7 (col 3)
    # C_13: C cols 4,5. Indices 8,9 (col 4), 10,11 (col 5)
    
    add_strassen(0, 0, 0)
    add_strassen(0, 2, 4)
    add_strassen(0, 4, 8)

    # Block 2: A_2 (2x2) cols 2,3. B_2 (2x6) rows 2,3.
    # A_2 indices: 4,5,6,7
    # B_2 indices: 12..17 (row 2), 18..23 (row 3)
    # B_21: cols 0,1. Indices 12,13,18,19
    # B_22: cols 2,3. Indices 14,15,20,21
    # B_23: cols 4,5. Indices 16,17,22,23
    # C_2: C cols 0..5.
    # C_21: C cols 0,1. Indices 0,1,2,3
    # C_22: C cols 2,3. Indices 4,5,6,7
    # C_23: C cols 4,5. Indices 8,9,10,11
    
    add_strassen(4, 12, 0)
    add_strassen(4, 14, 4)
    add_strassen(4, 16, 8)

    # Block 3: A_3 (2x1) col 4. B_3 (1x6) row 4.
    # A_3 indices: 8,9
    # B_3 indices: 24..29
    # C_3: C cols 0..5.
    # C_31: C col 0. Indices 0,1
    # ...
    # C_36: C col 5. Indices 10,11
    
    # C_ij = A_i B_j
    # A_3 is 2x1. B_3 is 1x6.
    # C is 2x6.
    # C_0j = A_3[0] * B_3[j]
    # C_1j = A_3[1] * B_3[j]
    
    for j in range(6):
        # Term for C_0j
        # U: A_3[0] (idx 8)
        # V: B_3[j] (idx 24+j)
        # W: C_0j (idx j*2 + 0)
        add_term([8], [1], [24+j], [1], [j*2], [1])
        
        # Term for C_1j
        # U: A_3[1] (idx 9)
        # V: B_3[j] (idx 24+j)
        # W: C_1j (idx j*2 + 1)
        add_term([9], [1], [24+j], [1], [j*2+1], [1])

    return {"U": U, "V": V, "W": W}
