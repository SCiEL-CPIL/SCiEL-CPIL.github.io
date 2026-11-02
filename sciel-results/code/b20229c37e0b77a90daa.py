"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 22
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_term(u_indices, u_vals, v_indices, v_vals, w_indices, w_vals, col):
        for idx, val in zip(u_indices, u_vals):
            if val != 0:
                U[idx][col] += val
        for idx, val in zip(v_indices, v_vals):
            if val != 0:
                V[idx][col] += val
        for idx, val in zip(w_indices, w_vals):
            if val != 0:
                W[idx][col] += val

    def add_strassen_222(a_idx, v_idx, w_idx, col_offset):
        # a_idx: [A00, A01, A10, A11]
        # v_idx: [B00, B01, B10, B11]
        # w_idx: [C00, C10, C01, C11] (Mapped to W indices)
        # Strassen Terms
        # M1 = (A00+A11)(B00+B11) -> C00+=M1, C11+=M1, C01-=M1, C10-=M1
        add_term([a_idx[0], a_idx[3]], [1, 1], [v_idx[0], v_idx[3]], [1, 1], 
                 [w_idx[0], w_idx[3], w_idx[2], w_idx[1]], [1, 1, -1, -1], col_offset)
        col_offset += 1
        # M2 = (A10+A11)B00 -> C10+=M2, C11+=M2
        add_term([a_idx[2], a_idx[3]], [1, 1], [v_idx[0]], [1], 
                 [w_idx[1], w_idx[3]], [1, 1], col_offset)
        col_offset += 1
        # M3 = (A00-A01)(B01+B11) -> C01+=M3, C11+=M3
        add_term([a_idx[0], a_idx[1]], [1, -1], [v_idx[1], v_idx[3]], [1, 1], 
                 [w_idx[2], w_idx[3]], [1, 1], col_offset)
        col_offset += 1
        # M4 = A11(B10-B11) -> C10+=M4, C11-=M4
        add_term([a_idx[3]], [1], [v_idx[2], v_idx[3]], [1, -1], 
                 [w_idx[1], w_idx[3]], [1, -1], col_offset)
        col_offset += 1
        # M5 = (A00+A10)B11 -> C00+=M5, C10+=M5
        add_term([a_idx[0], a_idx[2]], [1, 1], [v_idx[3]], [1], 
                 [w_idx[0], w_idx[1]], [1, 1], col_offset)
        col_offset += 1
        # M6 = (A01-A10)(B00+B10) -> C00+=M6, C10-=M6
        add_term([a_idx[1], a_idx[2]], [1, -1], [v_idx[0], v_idx[2]], [1, 1], 
                 [w_idx[0], w_idx[1]], [1, -1], col_offset)
        col_offset += 1
        # M7 = (A00-A11)(B01+B10) -> C01+=M7, C10-=M7
        add_term([a_idx[0], a_idx[3]], [1, -1], [v_idx[1], v_idx[2]], [1, 1], 
                 [w_idx[2], w_idx[1]], [1, -1], col_offset)
        col_offset += 1
        return col_offset

    def add_schoolbook_212(a_idx, v_idx, w_idx, col_offset):
        # a_idx: [A00, A10] (2x1)
        # v_idx: [B00, B01] (1x2)
        # w_idx: [C00, C10, C01, C11]
        # C00 += A00*B00
        add_term([a_idx[0]], [1], [v_idx[0]], [1], [w_idx[0]], [1], col_offset)
        col_offset += 1
        # C01 += A00*B01
        add_term([a_idx[0]], [1], [v_idx[1]], [1], [w_idx[2]], [1], col_offset)
        col_offset += 1
        # C10 += A10*B00
        add_term([a_idx[1]], [1], [v_idx[0]], [1], [w_idx[1]], [1], col_offset)
        col_offset += 1
        # C11 += A10*B01
        add_term([a_idx[1]], [1], [v_idx[1]], [1], [w_idx[3]], [1], col_offset)
        col_offset += 1
        return col_offset

    # A indices (6): 0,1,2 (row 0), 3,4,5 (row 1)
    # A_L (2x2): 0,1,3,4
    # A_R (2x1): 2,5
    
    # B indices (12): 
    # Row 0: 0,1,2,3
    # Row 1: 4,5,6,7
    # Row 2: 8,9,10,11
    
    # W indices (8):
    # C[0,0]=0, C[1,0]=1, C[0,1]=2, C[1,1]=3
    # C[0,2]=4, C[1,2]=5, C[0,3]=6, C[1,3]=7
    
    col = 0
    
    # Block 1: Cols 0,1
    # B_T (2x2): 0,1,4,5
    # B_B (1x2): 8,9
    # W (2x2): 0,1,2,3
    col = add_strassen_222([0,1,3,4], [0,1,4,5], [0,1,2,3], col)
    col = add_schoolbook_212([2,5], [8,9], [0,1,2,3], col)
    
    # Block 2: Cols 2,3
    # B_T (2x2): 2,3,6,7
    # B_B (1x2): 10,11
    # W (2x2): 4,5,6,7
    col = add_strassen_222([0,1,3,4], [2,3,6,7], [4,5,6,7], col)
    col = add_schoolbook_212([2,5], [10,11], [4,5,6,7], col)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
