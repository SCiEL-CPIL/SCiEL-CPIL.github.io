"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 4, 4
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)]  # (8, 28)
    V = [[0] * rank for _ in range(m * p)]  # (16, 28)
    W = [[0] * rank for _ in range(p * n)]  # (8, 28)
    
    # Strassen 2x2 x 2x2 decomposition
    # For matrices A (2x2), B (2x2), C = A*B
    # A indices: A[0,0]=a00, A[0,1]=a01, A[1,0]=a10, A[1,1]=a11
    # B indices: B[0,0]=b00, B[0,1]=b01, B[1,0]=b10, B[1,1]=b11
    # C indices: C[0,0]=c00, C[0,1]=c01, C[1,0]=c10, C[1,1]=c11
    #
    # Strassen products:
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b01-b11)
    # M4 = a11(b10-b00)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b01)
    # M7 = (a01-a11)(b10+b11)
    #
    # C:
    # c00 = M1 + M4 - M5 + M7
    # c01 = M3 + M5
    # c10 = M2 + M4
    # c11 = M1 - M2 + M3 + M6
    
    def add_strassen_block(offset, a_row_base, a_col_base, b_row_base, b_col_base, c_row_base, c_col_base):
        """
        Add a Strassen block for multiplying A[a_row_base:a_row_base+2, a_col_base:a_col_base+2]
        by B[b_row_base:b_row_base+2, b_col_base:b_col_base+2]
        contributing to C[c_row_base:c_row_base+2, c_col_base:c_col_base+2]
        
        U index for A element (i,j): i*m + j = i*4 + j
        V index for B element (i,j): i*p + j = i*4 + j
        W index for C element (i,j): j*n + i = j*2 + i  (trace convention: W[k*n+i, r] where k=col, i=row)
        """
        # A is 2x2 at rows a_row_base, a_row_base+1 and cols a_col_base, a_col_base+1
        # B is 2x2 at rows b_row_base, b_row_base+1 and cols b_col_base, b_col_base+1
        # C is 2x2 at rows c_row_base, c_row_base+1 and cols c_col_base, c_col_base+1
        
        # U indices for A elements
        a00_u = a_row_base * m + a_col_base
        a01_u = a_row_base * m + a_col_base + 1
        a10_u = (a_row_base + 1) * m + a_col_base
        a11_u = (a_row_base + 1) * m + a_col_base + 1
        
        # V indices for B elements
        b00_v = b_row_base * p + b_col_base
        b01_v = b_row_base * p + b_col_base + 1
        b10_v = (b_row_base + 1) * p + b_col_base
        b11_v = (b_row_base + 1) * p + b_col_base + 1
        
        # W indices for C elements (trace convention: W[k*n+i, r] where k=col, i=row)
        # C element at row i, col j has W index = j*n + i
        c00_w = c_col_base * n + c_row_base
        c01_w = (c_col_base + 1) * n + c_row_base
        c10_w = c_col_base * n + (c_row_base + 1)
        c11_w = (c_col_base + 1) * n + (c_row_base + 1)
        
        # M1 = (a00+a11)(b00+b11)
        U[a00_u][offset + 0] += 1
        U[a11_u][offset + 0] += 1
        V[b00_v][offset + 0] += 1
        V[b11_v][offset + 0] += 1
        W[c00_w][offset + 0] += 1
        W[c11_w][offset + 0] += 1
        
        # M2 = (a10+a11)b00
        U[a10_u][offset + 1] += 1
        U[a11_u][offset + 1] += 1
        V[b00_v][offset + 1] += 1
        W[c10_w][offset + 1] += 1
        W[c11_w][offset + 1] -= 1
        
        # M3 = a00(b01-b11)
        U[a00_u][offset + 2] += 1
        V[b01_v][offset + 2] += 1
        V[b11_v][offset + 2] -= 1
        W[c01_w][offset + 2] += 1
        W[c11_w][offset + 2] += 1
        
        # M4 = a11(b10-b00)
        U[a11_u][offset + 3] += 1
        V[b10_v][offset + 3] += 1
        V[b00_v][offset + 3] -= 1
        W[c00_w][offset + 3] += 1
        W[c10_w][offset + 3] += 1
        
        # M5 = (a00+a01)b11
        U[a00_u][offset + 4] += 1
        U[a01_u][offset + 4] += 1
        V[b11_v][offset + 4] += 1
        W[c00_w][offset + 4] -= 1
        W[c01_w][offset + 4] += 1
        
        # M6 = (a10-a00)(b00+b01)
        U[a10_u][offset + 5] += 1
        U[a00_u][offset + 5] -= 1
        V[b00_v][offset + 5] += 1
        V[b01_v][offset + 5] += 1
        W[c11_w][offset + 5] += 1
        
        # M7 = (a01-a11)(b10+b11)
        U[a01_u][offset + 6] += 1
        U[a11_u][offset + 6] -= 1
        V[b10_v][offset + 6] += 1
        V[b11_v][offset + 6] += 1
        W[c00_w][offset + 6] += 1
    
    # C_left = A_left * B_top_left + A_right * B_bottom_left
    # A_left: rows 0-1, cols 0-1 of A (2x4)
    # A_right: rows 0-1, cols 2-3 of A (2x4)
    # B_top_left: rows 0-1, cols 0-1 of B (4x4)
    # B_bottom_left: rows 2-3, cols 0-1 of B (4x4)
    # B_top_right: rows 0-1, cols 2-3 of B (4x4)
    # B_bottom_right: rows 2-3, cols 2-3 of B (4x4)
    # C_left: rows 0-1, cols 0-1 of C (2x4)
    # C_right: rows 0-1, cols 2-3 of C (2x4)
    
    # Block 1: A_left * B_top_left -> C_left
    add_strassen_block(0, 0, 0, 0, 0, 0, 0)
    
    # Block 2: A_right * B_bottom_left -> C_left
    add_strassen_block(7, 0, 2, 2, 0, 0, 0)
    
    # Block 3: A_left * B_top_right -> C_right
    add_strassen_block(14, 0, 0, 0, 2, 0, 2)
    
    # Block 4: A_right * B_bottom_right -> C_right
    add_strassen_block(21, 0, 2, 2, 2, 0, 2)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
