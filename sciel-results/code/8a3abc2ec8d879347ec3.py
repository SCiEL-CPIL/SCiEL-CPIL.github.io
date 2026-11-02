"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 102 decomposition using Strassen for 4x4x4 blocks
    # 2 blocks of 4 columns (rank 49 each) + 1 column (rank 4)
    rank = 102
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen 2x2 factors
    # M1: (A11+A22)(B11+B22)
    # M2: (A21+A22)B11
    # M3: A11(B12-B22)
    # M4: A22(B21-B11)
    # M5: (A11+A12)B22
    # M6: (A21-A11)(B11+B12)
    # M7: (A12-A22)(B21+B22)
    
    def add_strassen_block(base_col, col_offset):
        # col_offset is the starting column index in B (0-3 or 4-7)
        # We need to map the 2x2 block indices to the global indices
        # A is 4x4, B is 4x9 (but we treat 4 cols at a time)
        
        # Strassen multiplication of 4x4 matrices
        # Split A into 2x2 blocks of 2x2 matrices
        # A11 (0,0), A12 (0,1), A21 (1,0), A22 (1,1)
        
        # M1
        U[0*4+0][base_col] = 1; U[0*4+1][base_col] = 1; U[2*4+2][base_col] = 1; U[2*4+3][base_col] = 1
        V[0*9+col_offset][base_col] = 1; V[0*9+col_offset+1][base_col] = 1; V[2*9+col_offset+2][base_col] = 1; V[2*9+col_offset+3][base_col] = 1
        W[0*4+0][base_col] = 1; W[0*4+1][base_col] = 1; W[2*4+2][base_col] = 1; W[2*4+3][base_col] = 1
        base_col += 1
        
        # M2
        U[2*4+2][base_col] = 1; U[2*4+3][base_col] = 1
        V[0*9+col_offset][base_col] = 1; V[0*9+col_offset+1][base_col] = 1
        W[0*4+2][base_col] = 1; W[0*4+3][base_col] = 1
        base_col += 1
        
        # M3
        U[0*4+0][base_col] = 1; U[0*4+1][base_col] = 1
        V[0*9+col_offset+2][base_col] = 1; V[0*9+col_offset+3][base_col] = 1; V[2*9+col_offset+2][base_col] = -1; V[2*9+col_offset+3][base_col] = -1
        W[0*4+0][base_col] = 1; W[0*4+1][base_col] = 1
        base_col += 1
        
        # M4
        U[2*4+2][base_col] = 1; U[2*4+3][base_col] = 1
        V[2*9+col_offset][base_col] = 1; V[2*9+col_offset+1][base_col] = 1; V[0*9+col_offset][base_col] = -1; V[0*9+col_offset+1][base_col] = -1
        W[2*4+2][base_col] = 1; W[2*4+3][base_col] = 1
        base_col += 1
        
        # M5
        U[0*4+0][base_col] = 1; U[0*4+1][base_col] = 1; U[0*4+2][base_col] = 1; U[0*4+3][base_col] = 1
        V[2*9+col_offset+2][base_col] = 1; V[2*9+col_offset+3][base_col] = 1
        W[0*4+0][base_col] = 1; W[0*4+1][base_col] = 1
        base_col += 1
        
        # M6
        U[2*4+2][base_col] = 1; U[2*4+3][base_col] = 1; U[0*4+0][base_col] = -1; U[0*4+1][base_col] = -1
        V[0*9+col_offset][base_col] = 1; V[0*9+col_offset+1][base_col] = 1; V[0*9+col_offset+2][base_col] = 1; V[0*9+col_offset+3][base_col] = 1
        W[2*4+0][base_col] = 1; W[2*4+1][base_col] = 1
        base_col += 1
        
        # M7
        U[0*4+2][base_col] = 1; U[0*4+3][base_col] = 1; U[2*4+2][base_col] = -1; U[2*4+3][base_col] = -1
        V[2*9+col_offset][base_col] = 1; V[2*9+col_offset+1][base_col] = 1; V[2*9+col_offset+2][base_col] = 1; V[2*9+col_offset+3][base_col] = 1
        W[0*4+2][base_col] = 1; W[0*4+3][base_col] = 1
        base_col += 1
        
        return base_col

    # Block 1: cols 0-3
    col = add_strassen_block(0, 0)
    
    # Block 2: cols 4-7
    col = add_strassen_block(col, 4)
    
    # Block 3: col 8 (Schoolbook 4x4x1)
    for i in range(4):
        for j in range(4):
            U[i*4+j][col] = 1
            V[j*9+8][col] = 1
            W[8*4+i][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
