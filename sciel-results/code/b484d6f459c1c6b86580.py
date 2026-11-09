"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 34 decomposition using Strassen for 2x2x2 blocks
    rank = 34
    U = [[0] * rank for _ in range(n * m)] # 6 x 34
    V = [[0] * rank for _ in range(m * p)] # 18 x 34
    W = [[0] * rank for _ in range(p * n)] # 12 x 34
    
    # Helper to add a rank-1 term
    def add_term(u_idx, u_val, v_idx, v_val, w_idx, w_val, col):
        U[u_idx][col] = u_val
        V[v_idx][col] = v_val
        W[w_idx][col] = w_val

    # Block 1: Columns 0-2 of B, Columns 0-2 of C
    # Block 2: Columns 3-5 of B, Columns 3-5 of C
    
    # We define the Strassen terms for 2x2x2
    # A indices: 0,1,4,5
    # B indices: 0,1,3,4
    # C indices: 0,1,4,5
    
    def add_strassen_block(b_offset, c_offset, start_col):
        # b_offset: 0 or 3
        # c_offset: 0 or 3
        
        # M1: (A00+A11)(B00+B11) -> C00, C11
        add_term(0, 1, b_offset, 1, c_offset, 1, start_col + 0)
        add_term(5, 1, b_offset + 3, 1, c_offset + 3, 1, start_col + 0)
        
        # M2: (A10+A11)B00 -> C10, -C11
        add_term(4, 1, b_offset, 1, c_offset + 2, 1, start_col + 1)
        add_term(5, 1, b_offset, 1, c_offset + 3, -1, start_col + 1)
        
        # M3: A00(B01-B11) -> C01, C11
        add_term(0, 1, b_offset + 1, 1, c_offset + 1, 1, start_col + 2)
        add_term(0, 1, b_offset + 3, -1, c_offset + 3, 1, start_col + 2)
        
        # M4: A11(B10-B00) -> C00, C10
        add_term(5, 1, b_offset + 2, 1, c_offset, 1, start_col + 3)
        add_term(5, 1, b_offset, -1, c_offset + 2, 1, start_col + 3)
        
        # M5: (A00+A01)B11 -> -C00, C01
        add_term(0, 1, b_offset + 3, 1, c_offset, -1, start_col + 4)
        add_term(1, 1, b_offset + 3, 1, c_offset + 1, 1, start_col + 4)
        
        # M6: (A10-A00)(B00+B01) -> C11
        add_term(4, 1, b_offset, 1, c_offset + 3, 1, start_col + 5)
        add_term(0, -1, b_offset, 1, c_offset + 3, 1, start_col + 5)
        add_term(4, 1, b_offset + 1, 1, c_offset + 3, 1, start_col + 5)
        add_term(0, -1, b_offset + 1, 1, c_offset + 3, 1, start_col + 5)
        
        # M7: (A01-A11)(B10+B11) -> C00, -C01
        add_term(1, 1, b_offset + 2, 1, c_offset, 1, start_col + 6)
        add_term(5, -1, b_offset + 2, 1, c_offset, 1, start_col + 6)
        add_term(1, 1, b_offset + 3, 1, c_offset + 1, -1, start_col + 6)
        add_term(5, -1, b_offset + 3, 1, c_offset + 1, -1, start_col + 6)
        
        # a c terms (rank 4)
        # a: A[0,2]=2, A[1,2]=6
        # c: B[2,0]=6+b_offset, B[2,1]=7+b_offset
        # C: C[0,0]=c_offset, C[0,1]=c_offset+1, C[1,0]=c_offset+2, C[1,1]=c_offset+3
        
        # a0 c0 -> C00
        add_term(2, 1, 6 + b_offset, 1, c_offset, 1, start_col + 7)
        
        # a0 c1 -> C01
        add_term(2, 1, 7 + b_offset, 1, c_offset + 1, 1, start_col + 8)
        
        # a1 c0 -> C10
        add_term(6, 1, 6 + b_offset, 1, c_offset + 2, 1, start_col + 9)
        
        # a1 c1 -> C11
        add_term(6, 1, 7 + b_offset, 1, c_offset + 3, 1, start_col + 10)
        
        # A22 b terms (rank 4)
        # A: 0,1,4,5
        # b: B[0,2]=2+b_offset, B[1,2]=5+b_offset
        # C: C[0,2]=c_offset+2, C[1,2]=c_offset+5
        
        # A00 b0 -> C02
        add_term(0, 1, 2 + b_offset, 1, c_offset + 2, 1, start_col + 11)
        
        # A01 b1 -> C02
        add_term(1, 1, 5 + b_offset, 1, c_offset + 2, 1, start_col + 12)
        
        # A10 b0 -> C12
        add_term(4, 1, 2 + b_offset, 1, c_offset + 5, 1, start_col + 13)
        
        # A11 b1 -> C12
        add_term(5, 1, 5 + b_offset, 1, c_offset + 5, 1, start_col + 14)
        
        # a d terms (rank 2)
        # a: 2,6
        # d: B[2,2]=8+b_offset
        # C: C[0,2]=c_offset+2, C[1,2]=c_offset+5
        
        # a0 d -> C02
        add_term(2, 1, 8 + b_offset, 1, c_offset + 2, 1, start_col + 15)
        
        # a1 d -> C12
        add_term(6, 1, 8 + b_offset, 1, c_offset + 5, 1, start_col + 16)

    # Block 1
    add_strassen_block(0, 0, 0)
    # Block 2
    add_strassen_block(3, 3, 17)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
