DIMENSIONS = [2, 3, 8]

def construct():
    n, m, p = DIMENSIONS  # 2, 3, 8
    # A: 2x3, B: 3x8, C: 2x8
    # Split B into B_top (2x8) and B_bottom (1x8)
    # C = A_left(2x2) * B_top(2x8) + A_right(2x1) * B_bottom(1x8)
    # 
    # Part 1: 2x2x8 via two 2x2x4 blocks, each via two 2x2x2 Strassen blocks
    # Part 2: 2x1x8, rank 16
    #
    # Total rank: 28 + 16 = 44
    
    rank = 44
    U = [[0] * rank for _ in range(n * m)]  # 6 x 44
    V = [[0] * rank for _ in range(m * p)]  # 24 x 44
    W = [[0] * rank for _ in range(p * n)]  # 16 x 44
    
    # Indexing:
    # A[i,j]: i in 0..1, j in 0..2 -> U index i*3+j
    # B[j,k]: j in 0..2, k in 0..7 -> V index j*8+k
    # C[i,k]: i in 0..1, k in 0..7 -> W index k*2+i (trace convention W[k*n+i])
    
    # Part 2: A_right (column 2 of A) * B_bottom (row 2 of B)
    # A[0,2]*B[2,k] contributes to C[0,k]
    # A[1,2]*B[2,k] contributes to C[1,k]
    # This is 2x1x8, rank 16
    col = 0
    for i in range(2):  # row of A (and C)
        for k in range(8):  # column of B (and C)
            U[i * 3 + 2][col] = 1  # A[i,2]
            V[2 * 8 + k][col] = 1  # B[2,k]
            W[k * 2 + i][col] = 1  # C[i,k]
            col += 1
    
    # Part 1: A_left (2x2) * B_top (2x8)
    # A_left[i,j] for i in 0..1, j in 0..1 -> A[i,j]
    # B_top[j,k] for j in 0..1, k in 0..7 -> B[j,k]
    # Split B_top into two 2x4 blocks: B_top1 (k=0..3), B_top2 (k=4..7)
    # Each 2x2x4 via two 2x2x2 Strassen blocks
    
    # Strassen for 2x2x2:
    # A = [[a00,a01],[a10,a11]], B = [[b00,b01],[b10,b11]], C = [[c00,c01],[c10,c11]]
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b10-b11)
    # M4 = a11(b01-b00)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b10)
    # M7 = (a01-a11)(b01+b11)
    # c00 = M1+M4+M6-M7
    # c01 = M1+M5
    # c10 = M2+M3
    # c11 = M1+M3+M5+M7
    
    def strassen_block(base_col, a_row_offset, b_col_offset):
        """
        a_row_offset: which rows of A_left (0 or 1) - actually A_left is 2x2, so rows 0,1
        b_col_offset: which 2 columns of the 2x4 block
        """
        # A_left is A[0:2, 0:2], so A[i,j] for i,j in 0..1
        # B_top is B[0:2, b_col_offset:b_col_offset+2]
        
        a00_idx = 0 * 3 + 0  # A[0,0]
        a01_idx = 0 * 3 + 1  # A[0,1]
        a10_idx = 1 * 3 + 0  # A[1,0]
        a11_idx = 1 * 3 + 1  # A[1,1]
        
        b00_idx = 0 * 8 + b_col_offset      # B[0, b_col_offset]
        b01_idx = 0 * 8 + b_col_offset + 1  # B[0, b_col_offset+1]
        b10_idx = 1 * 8 + b_col_offset      # B[1, b_col_offset]
        b11_idx = 1 * 8 + b_col_offset + 1  # B[1, b_col_offset+1]
        
        c00_idx = b_col_offset * 2 + 0      # C[0, b_col_offset]
        c01_idx = (b_col_offset + 1) * 2 + 0  # C[0, b_col_offset+1]
        c10_idx = b_col_offset * 2 + 1      # C[1, b_col_offset]
        c11_idx = (b_col_offset + 1) * 2 + 1  # C[1, b_col_offset+1]
        
        # M1 = (a00+a11)(b00+b11)
        U[a00_idx][base_col] = 1
        U[a11_idx][base_col] = 1
        V[b00_idx][base_col] = 1
        V[b11_idx][base_col] = 1
        W[c00_idx][base_col] = 1
        W[c01_idx][base_col] = 1
        W[c11_idx][base_col] = 1
        base_col += 1
        
        # M2 = (a10+a11)b00
        U[a10_idx][base_col] = 1
        U[a11_idx][base_col] = 1
        V[b00_idx][base_col] = 1
        W[c10_idx][base_col] = 1
        W[c11_idx][base_col] = 1
        base_col += 1
        
        # M3 = a00(b10-b11)
        U[a00_idx][base_col] = 1
        V[b10_idx][base_col] = 1
        V[b11_idx][base_col] = -1
        W[c10_idx][base_col] = 1
        W[c11_idx][base_col] = 1
        base_col += 1
        
        # M4 = a11(b01-b00)
        U[a11_idx][base_col] = 1
        V[b01_idx][base_col] = 1
        V[b00_idx][base_col] = -1
        W[c00_idx][base_col] = 1
        W[c11_idx][base_col] = 1
        base_col += 1
        
        # M5 = (a00+a01)b11
        U[a00_idx][base_col] = 1
        U[a01_idx][base_col] = 1
        V[b11_idx][base_col] = 1
        W[c01_idx][base_col] = 1
        W[c11_idx][base_col] = 1
        base_col += 1
        
        # M6 = (a10-a00)(b00+b10)
        U[a10_idx][base_col] = 1
        U[a00_idx][base_col] = -1
        V[b00_idx][base_col] = 1
        V[b10_idx][base_col] = 1
        W[c00_idx][base_col] = -1
        W[c10_idx][base_col] = 1
        base_col += 1
        
        # M7 = (a01-a11)(b01+b11)
        U[a01_idx][base_col] = 1
        U[a11_idx][base_col] = -1
        V[b01_idx][base_col] = 1
        V[b11_idx][base_col] = 1
        W[c00_idx][base_col] = -1
        W[c01_idx][base_col] = 1
        base_col += 1
        
        return base_col
    
    # 2x2x4 block 1: B_top columns 0-3, split into two 2x2x2: cols 0-1 and cols 2-3
    col = strassen_block(col, 0, 0)   # cols 0-1
    col = strassen_block(col, 0, 2)   # cols 2-3
    
    # 2x2x4 block 2: B_top columns 4-7, split into two 2x2x2: cols 4-5 and cols 6-7
    col = strassen_block(col, 0, 4)   # cols 4-5
    col = strassen_block(col, 0, 6)   # cols 6-7
    
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    return {"U": U, "V": V, "W": W}
