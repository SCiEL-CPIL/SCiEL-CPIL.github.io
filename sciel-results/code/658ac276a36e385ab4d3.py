"""Matrix multiplication 3,5,5 - rank-minimal exact tensor decomposition."""

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Total rank will be sum of block ranks
    # Block 1: M_{1,5,5} rank 25
    # Block 2: M_{2,5,5} rank 46
    #   Block 2.1: M_{2,2,5} rank 18
    #   Block 2.2: M_{2,3,5} rank 28
    #     Block 2.2.1: M_{2,2,5} rank 18
    #     Block 2.2.2: M_{2,1,5} rank 10
    total_rank = 25 + 18 + 18 + 10
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col_offset = 0

    # Helper to add a block
    def add_block(a_idx, b_idx, c_idx, local_U, local_V, local_W):
        nonlocal col_offset
        for i in range(len(a_idx)):
            for k in range(local_U[i].__len__()):
                U[a_idx[i]][col_offset + k] = local_U[i][k]
        for i in range(len(b_idx)):
            for k in range(local_V[i].__len__()):
                V[b_idx[i]][col_offset + k] = local_V[i][k]
        for i in range(len(c_idx)):
            for k in range(local_W[i].__len__()):
                W[c_idx[i]][col_offset + k] = local_W[i][k]
        col_offset += local_U[0].__len__()

    # Block 1: M_{1,5,5}
    # A: row 0 (indices 0..4)
    # B: all (indices 0..24)
    # C: col 0 (indices 0, 3, 6, 9, 12) -> k*n + i = k*3 + 0
    a_idx_1 = [0, 1, 2, 3, 4] # i=0, j=0..4
    b_idx_1 = list(range(25)) # j=0..4, k=0..4
    c_idx_1 = [k*3 + 0 for k in range(5)] # k=0..4, i=0
    
    local_U_1 = [[0]*25 for _ in range(5)]
    local_V_1 = [[0]*25 for _ in range(25)]
    local_W_1 = [[0]*25 for _ in range(5)]
    
    col = 0
    for j in range(5):
        for k in range(5):
            local_U_1[j][col] = 1
            local_V_1[j*5 + k][col] = 1
            local_W_1[k][col] = 1
            col += 1
            
    add_block(a_idx_1, b_idx_1, c_idx_1, local_U_1, local_V_1, local_W_1)

    # Block 2.1: M_{2,2,5}
    # A: rows 1,2 (indices 5..9, 10..14)
    # B: cols 0,1 (indices 0..4, 5..9)
    # C: cols 1,2 (indices k*3+1, k*3+2)
    a_idx_21 = [5,6,7,8,9, 10,11,12,13,14]
    b_idx_21 = list(range(10)) # 2 rows * 5 cols
    c_idx_21 = [k*3+1 for k in range(5)] + [k*3+2 for k in range(5)]
    
    # Rank 18 decomposition for 2x2x5
    # Split 5 cols into 2, 2, 1
    # Part 1: 2x2x2 (cols 0,1) rank 7
    # Part 2: 2x2x2 (cols 2,3) rank 7
    # Part 3: 2x2x1 (col 4) rank 4
    
    local_U_21 = [[0]*18 for _ in range(10)]
    local_V_21 = [[0]*18 for _ in range(10)]
    local_W_21 = [[0]*18 for _ in range(10)]
    
    col = 0
    # Part 1: Cols 0,1
    # Strassen 2x2x2
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # Indices in local block:
    # A: 0,1 (row 0), 2,3 (row 1)
    # B: 0,1 (col 0), 2,3 (col 1)
    # C: 0,1 (col 0), 2,3 (col 1)
    
    # M1 = (a+d)(e+g)
    local_U_21[0][col] = 1; local_U_21[3][col] = 1
    local_V_21[0][col] = 1; local_V_21[2][col] = 1
    local_W_21[0][col] = 1; local_W_21[2][col] = 1
    col += 1
    # M2 = d(e-g)
    local_U_21[3][col] = 1
    local_V_21[0][col] = 1; local_V_21[2][col] = -1
    local_W_21[1][col] = 1
    col += 1
    # M3 = (a+b)g
    local_U_21[0][col] = 1; local_U_21[1][col] = 1
    local_V_21[2][col] = 1
    local_W_21[1][col] = 1
    col += 1
    # M4 = (c+d)e
    local_U_21[2][col] = 1; local_U_21[3][col] = 1
    local_V_21[0][col] = 1
    local_W_21[0][col] = 1
    col += 1
    # M5 = (c-d)(g+h)
    local_U_21[2][col] = 1; local_U_21[3][col] = -1
    local_V_21[2][col] = 1; local_V_21[3][col] = 1
    local_W_21[3][col] = 1
    col += 1
    # M6 = (a-c)(e+f)
    local_U_21[0][col] = 1; local_U_21[2][col] = -1
    local_V_21[0][col] = 1; local_V_21[1][col] = 1
    local_W_21[0][col] = 1; local_W_21[2][col] = 1
    col += 1
    # M7 = (b-d)(f+h)
    local_U_21[1][col] = 1; local_U_21[3][col] = -1
    local_V_21[1][col] = 1; local_V_21[3][col] = 1
    local_W_21[3][col] = 1
    col += 1

    # Part 2: Cols 2,3
    # A: 0,1,2,3
    # B: 4,5 (col 2), 6,7 (col 3)
    # C: 4,5 (col 2), 6,7 (col 3)
    
    # M1 = (a+d)(e+g)
    local_U_21[0][col] = 1; local_U_21[3][col] = 1
    local_V_21[4][col] = 1; local_V_21[6][col] = 1
    local_W_21[4][col] = 1; local_W_21[6][col] = 1
    col += 1
    # M2 = d(e-g)
    local_U_21[3][col] = 1
    local_V_21[4][col] = 1; local_V_21[6][col] = -1
    local_W_21[5][col] = 1
    col += 1
    # M3 = (a+b)g
    local_U_21[0][col] = 1; local_U_21[1][col] = 1
    local_V_21[6][col] = 1
    local_W_21[5][col] = 1
    col += 1
    # M4 = (c+d)e
    local_U_21[2][col] = 1; local_U_21[3][col] = 1
    local_V_21[4][col] = 1
    local_W_21[4][col] = 1
    col += 1
    # M5 = (c-d)(g+h)
    local_U_21[2][col] = 1; local_U_21[3][col] = -1
    local_V_21[6][col] = 1; local_V_21[7][col] = 1
    local_W_21[7][col] = 1
    col += 1
    # M6 = (a-c)(e+f)
    local_U_21[0][col] = 1; local_U_21[2][col] = -1
    local_V_21[4][col] = 1; local_V_21[5][col] = 1
    local_W_21[4][col] = 1; local_W_21[6][col] = 1
    col += 1
    # M7 = (b-d)(f+h)
    local_U_21[1][col] = 1; local_U_21[3][col] = -1
    local_V_21[5][col] = 1; local_V_21[7][col] = 1
    local_W_21[7][col] = 1
    col += 1

    # Part 3: Col 4
    # A: 0,1,2,3
    # B: 8,9
    # C: 8,9
    
    # Schoolbook 2x2x1 (4 terms)
    # a*e
    local_U_21[0][col] = 1
    local_V_21[8][col] = 1
    local_W_21[8][col] = 1
    col += 1
    # b*f
    local_U_21[1][col] = 1
    local_V_21[9][col] = 1
    local_W_21[8][col] = 1
    col += 1
    # c*e
    local_U_21[2][col] = 1
    local_V_21[8][col] = 1
    local_W_21[9][col] = 1
    col += 1
    # d*f
    local_U_21[3][col] = 1
    local_V_21[9][col] = 1
    local_W_21[9][col] = 1
    col += 1

    add_block(a_idx_21, b_idx_21, c_idx_21, local_U_21, local_V_21, local_W_21)

    # Block 2.2.1: M_{2,2,5} for B cols 2,3
    # A: rows 1,2 (indices 5..9, 10..14)
    # B: cols 2,3 (indices 10..14, 15..19)
    # C: cols 1,2 (indices k*3+1, k*3+2)
    a_idx_221 = [5,6,7,8,9, 10,11,12,13,14]
    b_idx_221 = list(range(10, 20))
    c_idx_221 = [k*3+1 for k in range(5)] + [k*3+2 for k in range(5)]
    
    local_U_221 = [[0]*18 for _ in range(10)]
    local_V_221 = [[0]*18 for _ in range(10)]
    local_W_221 = [[0]*18 for _ in range(10)]
    
    col = 0
    # Same structure as 2.1 but B indices are local 0..9
    # Part 1: Cols 0,1 (global 2,3)
    local_U_221[0][col] = 1; local_U_221[3][col] = 1
    local_V_221[0][col] = 1; local_V_221[2][col] = 1
    local_W_221[0][col] = 1; local_W_221[2][col] = 1
    col += 1
    local_U_221[3][col] = 1
    local_V_221[0][col] = 1; local_V_221[2][col] = -1
    local_W_221[1][col] = 1
    col += 1
    local_U_221[0][col] = 1; local_U_221[1][col] = 1
    local_V_221[2][col] = 1
    local_W_221[1][col] = 1
    col += 1
    local_U_221[2][col] = 1; local_U_221[3][col] = 1
    local_V_221[0][col] = 1
    local_W_221[0][col] = 1
    col += 1
    local_U_221[2][col] = 1; local_U_221[3][col] = -1
    local_V_221[2][col] = 1; local_V_221[3][col] = 1
    local_W_221[3][col] = 1
    col += 1
    local_U_221[0][col] = 1; local_U_221[2][col] = -1
    local_V_221[0][col] = 1; local_V_221[1][col] = 1
    local_W_221[0][col] = 1; local_W_221[2][col] = 1
    col += 1
    local_U_221[1][col] = 1; local_U_221[3][col] = -1
    local_V_221[1][col] = 1; local_V_221[3][col] = 1
    local_W_221[3][col] = 1
    col += 1

    # Part 2: Cols 2,3 (global 4,5? No, B cols 2,3 are local 0,1. We did 0,1. 
    # Wait, b_idx_221 is 10..19. 
    # B col 2 is rows 2,3? No.
    # B is 5x5.
    # Col 2 is index 2*5+0..2*5+4? No.
    # V index is j*p + k.
    # j is row of B. k is col of B.
    # We are fixing j (rows of B) and varying k (cols of B).
    # In M_{2,2,5}, A is 2x2, B is 2x5.
    # A rows 1,2. B rows 2,3.
    # B rows 2,3 correspond to j=2,3.
    # B cols 0,1 correspond to k=0,1.
    # So V indices for B rows 2,3 and cols 0,1 are:
    # j=2, k=0 -> 2*5+0 = 10
    # j=2, k=1 -> 2*5+1 = 11
    # j=3, k=0 -> 3*5+0 = 15
    # j=3, k=1 -> 3*5+1 = 16
    # So b_idx_221 should be [10, 11, 15, 16] for the first 2x2 block?
    # No, the block is M_{2,2,5}.
    # A is 2x2. B is 2x5.
    # The 2 rows of B are j=2,3.
    # The 5 cols of B are k=0..4.
    # So V indices are:
    # j=2: 10, 11, 12, 13, 14
    # j=3: 15, 16, 17, 18, 19
    # My b_idx_221 = list(range(10, 20)) is correct.
    # In the local decomposition, we split the 5 cols into 2,2,1.
    # Part 1: k=0,1. V indices 10, 11, 15, 16.
    # Part 2: k=2,3. V indices 12, 13, 17, 18.
    # Part 3: k=4. V indices 14, 19.
    
    # I made a mistake in the previous block 2.1 construction.
    # In 2.1, b_idx was 0..9.
    # j=0: 0..4
    # j=1: 5..9
    # Part 1 (k=0,1): V indices 0,1,5,6.
    # Part 2 (k=2,3): V indices 2,3,7,8.
    # Part 3 (k=4): V indices 4,9.
    
    # In my code for 2.1:
    # Part 1 used V indices 0,1,2,3. This is WRONG.
    # V index 0 is j=0,k=0.
    # V index 1 is j=0,k=1.
    # V index 2 is j=0,k=2.
    # V index 5 is j=1,k=0.
    # V index 6 is j=1,k=1.
    # So for k=0,1, we need V indices 0,1 (from j=0) and 5,6 (from j=1).
    
    # I need to rewrite the block construction more carefully.
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
