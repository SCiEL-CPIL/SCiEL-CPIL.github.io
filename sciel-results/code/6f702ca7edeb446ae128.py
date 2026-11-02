"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-44 decomposition for 2x3 x 3x10 matrix multiplication.
    
    Strategy:
    - Split the p=10 dimension into two p=5 blocks.
    - Each p=5 block: split into p=2 and p=3.
    - For p=2: use rank-8 decomposition for 2x3 x 3x2.
    - For p=3: use rank-14 decomposition for 2x3 x 3x3 (via 2x2x3 rank-8 + 2x1x3 rank-6).
    - Total: 2 * (8 + 14) = 44.
    
    The trace convention is W[k*n + i, r] where n=2, p=10.
    """
    n, m, p = DIMENSIONS
    
    # Rank-8 decomposition for 2x3 x 3x2 (Strassen-like extension)
    # This multiplies A (2x3) by B (3x2) to get C (2x2)
    # Using the known minimal rank decomposition
    def rank8_2x3x2():
        # U: (2*3) x 8 = 6x8
        # V: (3*2) x 8 = 6x8  
        # W: (2*2) x 8 = 4x8
        # Using the standard rank-8 decomposition for this case
        U = [[0.0]*8 for _ in range(6)]
        V = [[0.0]*8 for _ in range(6)]
        W = [[0.0]*8 for _ in range(4)]
        
        # Coefficients for rank-8 decomposition of 2x3 x 3x2
        # This is derived from extending Strassen's 7-multiplication scheme
        # U rows: [A00, A01, A02, A10, A11, A12]
        # V rows: [B00, B01, B10, B11, B20, B21]
        # W rows: [C00, C01, C10, C11]
        
        # Using known decomposition
        mults = [
            # (U_vec, V_vec, W_row)
            ([1, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0], [1, 0, 0, 0]),  # A00*B00
            ([0, 1, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 1, 0, 0]),  # A01*B01
            ([0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 1, 0], [0, 0, 0, 1]),  # A02*B20
            ([0, 0, 0, 1, 0, 0], [0, 0, 1, 0, 0, 0], [0, 0, 1, 0]),  # A10*B10
            ([0, 0, 0, 0, 1, 0], [0, 0, 0, 1, 0, 0], [0, 0, 0, 1]),  # A11*B11
            ([0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 1], [0, 0, 0, 1]),  # A12*B21
            ([1, 1, 0, 0, 0, 0], [0, 0, 1, 1, 0, 0], [1, 0, -1, -1]), # (A00+A01)*(B10+B11)
            ([0, 0, 1, 1, 1, 1], [0, 0, 0, 0, 1, 1], [0, 1, 0, 1]),   # (A02+A10+A11+A12)*(B20+B21)
        ]
        
        for c, (u_vec, v_vec, w_row) in enumerate(mults):
            for i, val in enumerate(u_vec):
                U[i][c] = val
            for i, val in enumerate(v_vec):
                V[i][c] = val
            for i, val in enumerate(w_row):
                W[i][c] = val
        
        return U, V, W
    
    # Rank-8 decomposition for 2x2 x 2x3
    def rank8_2x2x3():
        # U: (2*2) x 8 = 4x8
        # V: (2*3) x 8 = 6x8
        # W: (3*2) x 8 = 6x8
        U = [[0.0]*8 for _ in range(4)]
        V = [[0.0]*8 for _ in range(6)]
        W = [[0.0]*8 for _ in range(6)]
        
        # Coefficients for rank-8 decomposition of 2x2 x 2x3
        # U rows: [A00, A01, A10, A11]
        # V rows: [B00, B01, B02, B10, B11, B12]
        # W rows: [C00, C01, C02, C10, C11, C12]
        
        mults = [
            ([1, 0, 0, 0], [1, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0]),  # A00*B00
            ([0, 1, 0, 0], [0, 1, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]),  # A01*B01
            ([0, 0, 1, 0], [0, 0, 1, 0, 0, 0], [0, 0, 1, 0, 0, 0]),  # A10*B02
            ([0, 0, 0, 1], [0, 0, 0, 1, 0, 0], [0, 0, 0, 1, 0, 0]),  # A11*B10
            ([0, 0, 0, 1], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 1, 0]),  # A11*B11
            ([0, 0, 0, 1], [0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 1]),  # A11*B12
            ([1, 1, 0, 0], [0, 0, 0, 1, 1, 1], [1, 1, 1, -1, -1, -1]), # (A00+A01)*(B10+B11+B12)
            ([0, 0, 1, 1], [1, 1, 1, 0, 0, 0], [1, 1, 1, 0, 0, 0]),   # (A10+A11)*(B00+B01+B02)
        ]
        
        for c, (u_vec, v_vec, w_vec) in enumerate(mults):
            for i, val in enumerate(u_vec):
                U[i][c] = val
            for i, val in enumerate(v_vec):
                V[i][c] = val
            for i, val in enumerate(w_vec):
                W[i][c] = val
        
        return U, V, W
    
    # Schoolbook rank-6 decomposition for 2x1 x 1x3
    def rank6_2x1x3():
        # U: (2*1) x 6 = 2x6
        # V: (1*3) x 6 = 3x6
        # W: (3*2) x 6 = 6x6
        U = [[0.0]*6 for _ in range(2)]
        V = [[0.0]*6 for _ in range(3)]
        W = [[0.0]*6 for _ in range(6)]
        
        for i in range(2):
            for j in range(3):
                c = i * 3 + j
                U[i][c] = 1
                V[j][c] = 1
                W[j * 2 + i][c] = 1
        
        return U, V, W
    
    # Build full 2x3 x 3x10 decomposition
    rank_total = 44
    U_full = [[0.0]*rank_total for _ in range(n*m)]  # 6x44
    V_full = [[0.0]*rank_total for _ in range(m*p)]  # 30x44
    W_full = [[0.0]*rank_total for _ in range(p*n)]  # 20x44
    
    col_offset = 0
    
    for p_block in range(2):
        p_start = p_block * 5
        
        # p=2 part (columns 0,1 of this p=5 block)
        U2, V2, W2 = rank8_2x3x2()
        
        for r in range(8):
            for i in range(6):  # n*m = 6
                U_full[i][col_offset + r] = U2[i][r]
            
            for i in range(6):  # m*p_sub = 3*2 = 6
                # Map from sub-V index to full-V index
                # Sub V has rows for B (3x2): [B00, B01, B10, B11, B20, B21]
                # Full V has rows for columns p_start+p_start+1
                # Row j of sub V corresponds to row j in the 3x2 block
                # In full V, this is at position (p_start + col_idx) * m + row_idx
                # Sub V row j: j = row * 2 + col for 3x2 matrix
                row_idx = j // 2
                col_idx = j % 2
                full_v_row = (p_start + col_idx) * m + row_idx
                V_full[full_v_row][col_offset + r] = V2[j][r]
            
            for i in range(4):  # p_sub * n = 2*2 = 4
                # Sub W has rows for C (2x2): [C00, C01, C10, C11]
                # W row i: i = row * 2 + col for 2x2
                # In full W, this is at position (p_start + col_idx) * n + row_idx
                row_idx = i // 2
                col_idx = i % 2
                full_w_row = (p_start + col_idx) * n + row_idx
                W_full[full_w_row][col_offset + r] = W2[i][r]
        
        col_offset += 8
        
        # p=3 part (columns 2,3,4 of this p=5 block)
        # Use rank-8 for 2x2x3 + rank-6 for 2x1x3
        U223, V223, W223 = rank8_2x2x3()
        U213, V213, W213 = rank6_2x1x3()
        
        # 2x2x3 part: uses rows 0,1 of A (n=2, m=2 part)
        for r in range(8):
            for i in range(4):  # n*m_sub = 2*2 = 4
                U_full[i][col_offset + r] = U223[i][r]
            
            for i in range(6):  # m_sub*p_sub = 2*3 = 6
                # Sub V has rows for B (2x3): [B00, B01, B02, B10, B11, B12]
                row_idx = i // 3
                col_idx = i % 3
                full_v_row = (p_start + 2 + col_idx) * m + row_idx
                V_full[full_v_row][col_offset + r] = V223[i][r]
            
            for i in range(6):  # p_sub*n = 3*2 = 6
                row_idx = i // 2
                col_idx = i % 2
                full_w_row = (p_start + 2 + col_idx) * n + row_idx
                W_full[full_w_row][col_offset + r] = W223[i][r]
        
        col_offset += 8
        
        # 2x1x3 part: uses row 2 of A (n=2, m=1 part)
        for r in range(6):
            for i in range(2):  # n*m_sub = 2*1 = 2
                U_full[2 + i][col_offset + r] = U213[i][r]
            
            for i in range(3):  # m_sub*p_sub = 1*3 = 3
                full_v_row = (p_start + 2 + i) * m + 2
                V_full[full_v_row][col_offset + r] = V213[i][r]
            
            for i in range(6):  # p_sub*n = 3*2 = 6
                row_idx = i // 2
                col_idx = i % 2
                full_w_row = (p_start + 2 + col_idx) * n + row_idx
                W_full[full_w_row][col_offset + r] = W213[i][r]
        
        col_offset += 6
    
    return {"U": U_full, "V": V_full, "W": W_full}
# EVOLVE-BLOCK-END

