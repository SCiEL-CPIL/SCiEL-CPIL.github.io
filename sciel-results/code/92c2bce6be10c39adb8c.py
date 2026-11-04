"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 4, 4
    rank = 28
    
    U = [[0]*rank for _ in range(n*m)]  # 8 x 28
    V = [[0]*rank for _ in range(m*p)]  # 16 x 28
    W = [[0]*rank for _ in range(p*n)]  # 8 x 28
    
    def add_strassen_block(A_row_off, A_col_off, B_row_off, B_col_off, C_row_off, C_col_off, col_start):
        a00 = A_row_off*4 + A_col_off
        a01 = A_row_off*4 + A_col_off + 1
        a10 = (A_row_off+1)*4 + A_col_off
        a11 = (A_row_off+1)*4 + A_col_off + 1
        
        b00 = B_row_off*4 + B_col_off
        b01 = B_row_off*4 + B_col_off + 1
        b10 = (B_row_off+1)*4 + B_col_off
        b11 = (B_row_off+1)*4 + B_col_off + 1
        
        c00 = C_col_off*2 + C_row_off
        c01 = (C_col_off+1)*2 + C_row_off
        c10 = C_col_off*2 + C_row_off + 1
        c11 = (C_col_off+1)*2 + C_row_off + 1
        
        col = col_start
        
        # M1 = (a00+a11)*(b00+b11)
        U[a00][col] += 1; U[a11][col] += 1
        V[b00][col] += 1; V[b11][col] += 1
        W[c00][col] += 1; W[c11][col] += 1
        col += 1
        
        # M2 = (a10+a11)*b00
        U[a10][col] += 1; U[a11][col] += 1
        V[b00][col] += 1
        W[c10][col] += 1; W[c11][col] += 1
        col += 1
        
        # M3 = a00*(b01-b11)
        U[a00][col] += 1
        V[b01][col] += 1; V[b11][col] -= 1
        W[c01][col] += 1; W[c11][col] += 1
        col += 1
        
        # M4 = a11*(b10-b00)
        U[a11][col] += 1
        V[b10][col] += 1; V[b00][col] -= 1
        W[c00][col] -= 1; W[c10][col] += 1
        col += 1
        
        # M5 = (a00+a01)*b11
        U[a00][col] += 1; U[a01][col] += 1
        V[b11][col] += 1
        W[c01][col] += 1; W[c11][col] += 1
        col += 1
        
        # M6 = (a10-a00)*(b00+b01)
        U[a10][col] += 1; U[a00][col] -= 1
        V[b00][col] += 1; V[b01][col] += 1
        W[c11][col] += 1
        col += 1
        
        # M7 = (a01-a11)*(b10+b11)
        U[a01][col] += 1; U[a11][col] -= 1
        V[b10][col] += 1; V[b11][col] += 1
        W[c00][col] += 1; W[c10][col] -= 1
        col += 1
        
        return col
    
    col = add_strassen_block(0, 0, 0, 0, 0, 0, 0)
    col = add_strassen_block(0, 2, 2, 0, 0, 0, col)
    col = add_strassen_block(0, 0, 0, 2, 0, 2, col)
    col = add_strassen_block(0, 2, 2, 2, 0, 2, col)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
