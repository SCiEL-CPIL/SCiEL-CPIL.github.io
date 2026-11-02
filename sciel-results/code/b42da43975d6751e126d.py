import numpy as np

def construct():
    n, m, p = 3, 3, 8
    
    # Total rank
    # M_2,2,8: 4 * 7 = 28
    # M_2,1,8: 2 * 8 = 16
    # M_1,2,8: 2 * 8 = 16
    # M_1,1,8: 1 * 8 = 8
    total_rank = 28 + 16 + 16 + 8
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a term
    def add_term(u_indices, v_indices, w_indices, u_vals, v_vals, w_vals):
        nonlocal col
        for idx, val in zip(u_indices, u_vals):
            U[idx][col] += val
        for idx, val in zip(v_indices, v_vals):
            V[idx][col] += val
        for idx, val in zip(w_indices, w_vals):
            W[idx][col] += val
        col += 1

    # 1. M_2,2,8 (A_11 * B_11)
    # A_11 is rows 0,1 of A, cols 0,1. Indices 0,1,3,4 in U.
    # B_11 is rows 0,1 of B, cols 0..7. Indices 0..7, 8..15 in V.
    # C_11 is rows 0,1 of C, cols 0..7. Indices 0..7, 8..15 in W (k*n+i).
    # W index for C_11[k, i] is k*3 + i.
    # C_11 is 2x8. k in 0..7, i in 0..1.
    
    # Split B_11 into 4 blocks of 2x2.
    # Block 0: cols 0,1
    # Block 1: cols 2,3
    # Block 2: cols 4,5
    # Block 3: cols 6,7
    
    for b_idx in range(4):
        # B indices for this block
        # Row 0: 2*b_idx, 2*b_idx+1
        # Row 1: 8+2*b_idx, 8+2*b_idx+1
        b_r0_c0 = 2 * b_idx
        b_r0_c1 = 2 * b_idx + 1
        b_r1_c0 = 8 + 2 * b_idx
        b_r1_c1 = 8 + 2 * b_idx + 1
        
        # W indices for C_11 block
        # C_11 is 2x8.
        # Row 0, Col 0: k=2*b_idx, i=0 -> W[2*b_idx*3 + 0]
        # Row 0, Col 1: k=2*b_idx+1, i=0 -> W[(2*b_idx+1)*3 + 0]
        # Row 1, Col 0: k=2*b_idx, i=1 -> W[2*b_idx*3 + 1]
        # Row 1, Col 1: k=2*b_idx+1, i=1 -> W[(2*b_idx+1)*3 + 1]
        
        w_r0_c0 = 2 * b_idx * 3 + 0
        w_r0_c1 = (2 * b_idx + 1) * 3 + 0
        w_r1_c0 = 2 * b_idx * 3 + 1
        w_r1_c1 = (2 * b_idx + 1) * 3 + 1
        
        # A indices for A_11
        # A_11 is 2x2.
        # (0,0) -> U[0]
        # (0,1) -> U[1]
        # (1,0) -> U[3]
        # (1,1) -> U[4]
        a00, a01, a10, a11 = 0, 1, 3, 4
        
        # Strassen terms
        # M1 = (a00+a11)(b00+b11)
        add_term([a00, a11], [b_r0_c0, b_r1_c1], [w_r0_c0, w_r1_c1], [1, 1], [1, 1], [1, 1])
        # M2 = (a10+a11)b00
        add_term([a10, a11], [b_r0_c0], [w_r1_c0, w_r1_c1], [1, 1], [1], [1, -1])
        # M3 = a00(b01-b11)
        add_term([a00], [b_r0_c1, b_r1_c1], [w_r0_c1, w_r1_c1], [1], [1, -1], [1, 1])
        # M4 = a11(b10-b00)
        add_term([a11], [b_r1_c0, b_r0_c0], [w_r0_c0, w_r1_c0], [1], [1, -1], [1, 1])
        # M5 = (a00+a01)b11
        add_term([a00, a01], [b_r1_c1], [w_r0_c0, w_r0_c1], [1, 1], [1], [-1, 1])
        # M6 = (a10-a00)(b00+b01)
        add_term([a10, a00], [b_r0_c0, b_r0_c1], [w_r1_c1], [1, -1], [1, 1], [1])
        # M7 = (a01-a11)(b10+b11)
        add_term([a01, a11], [b_r1_c0, b_r1_c1], [w_r0_c0, w_r1_c1], [1, -1], [1, 1], [1, 1])
        
    # 2. M_2,1,8 (A_12 * B_21)
    # A_12 is 2x1. Rows 0,1 of A, Col 2. Indices 2, 5 in U.
    # B_21 is 1x8. Row 2 of B, Cols 0..7. Indices 16..23 in V.
    # C_12 is 2x8. Rows 0,1 of C, Cols 0..7.
    # W indices: k*3 + i. i in 0,1. k in 0..7.
    
    a_12_r0, a_12_r1 = 2, 5
    
    for k in range(8):
        b_idx = 16 + k
        w_r0 = k * 3 + 0
        w_r1 = k * 3 + 1
        
        # Outer product: C[i,k] = A_12[i] * B_21[k]
        # Term for i=0:
        add_term([a_12_r0], [b_idx], [w_r0], [1], [1], [1])
        # Term for i=1:
        add_term([a_12_r1], [b_idx], [w_r1], [1], [1], [1])
        
    # 3. M_1,2,8 (A_21 * B_11)
    # A_21 is 1x2. Row 2 of A, Cols 0,1. Indices 6, 7 in U.
    # B_11 is 2x8. Rows 0,1 of B, Cols 0..7. Indices 0..7, 8..15 in V.
    # C_21 is 1x8. Row 2 of C, Cols 0..7.
    # W indices: k*3 + 2.
    
    a_21_c0, a_21_c1 = 6, 7
    
    for k in range(8):
        b_r0_idx = k
        b_r1_idx = 8 + k
        w_idx = k * 3 + 2
        
        # C[2,k] = A_21[0] * B_11[0,k] + A_21[1] * B_11[1,k]
        # Term 1:
        add_term([a_21_c0], [b_r0_idx], [w_idx], [1], [1], [1])
        # Term 2:
        add_term([a_21_c1], [b_r1_idx], [w_idx], [1], [1], [1])
        
    # 4. M_1,1,8 (A_22 * B_21)
    # A_22 is 1x1. Row 2, Col 2 of A. Index 8 in U.
    # B_21 is 1x8. Row 2 of B, Cols 0..7. Indices 16..23 in V.
    # C_22 is 1x8. Row 2 of C, Cols 0..7.
    # W indices: k*3 + 2.
    
    a_22 = 8
    
    for k in range(8):
        b_idx = 16 + k
        w_idx = k * 3 + 2
        
        # C[2,k] += A_22 * B_21[k]
        add_term([a_22], [b_idx], [w_idx], [1], [1], [1])
        
    return {"U": U, "V": V, "W": W}
