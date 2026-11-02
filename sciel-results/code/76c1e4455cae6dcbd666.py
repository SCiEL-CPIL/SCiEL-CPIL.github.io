"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Rank 36 decomposition
    rank = 36
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen(a_row_offset, a_col_offset, b_row_offset, b_col_offset, c_row_offset, c_col_offset):
        nonlocal col
        # A is 2x2 at (a_row_offset, a_col_offset)
        # B is 2x2 at (b_row_offset, b_col_offset)
        # C is 2x2 at (c_row_offset, c_col_offset)
        
        u00 = a_row_offset * m + a_col_offset
        u01 = a_row_offset * m + a_col_offset + 1
        u10 = (a_row_offset + 1) * m + a_col_offset
        u11 = (a_row_offset + 1) * m + a_col_offset + 1
        
        v00 = b_row_offset * p + b_col_offset
        v01 = b_row_offset * p + b_col_offset + 1
        v10 = (b_row_offset + 1) * p + b_col_offset
        v11 = (b_row_offset + 1) * p + b_col_offset + 1
        
        w00 = c_col_offset * n + c_row_offset
        w01 = (c_col_offset + 1) * n + c_row_offset
        w10 = c_col_offset * n + c_row_offset + 1
        w11 = (c_col_offset + 1) * n + c_row_offset + 1
        
        # M1 = (a00+a11)(b00+b11)
        U[u00][col] += 1; U[u11][col] += 1
        V[v00][col] += 1; V[v11][col] += 1
        W[w00][col] += 1; W[w11][col] += 1
        col += 1
        
        # M2 = (a10+a11)b00
        U[u10][col] += 1; U[u11][col] += 1
        V[v00][col] += 1
        W[w10][col] += 1; W[w11][col] -= 1
        col += 1
        
        # M3 = a00(b01-b11)
        U[u00][col] += 1
        V[v01][col] += 1; V[v11][col] -= 1
        W[w01][col] += 1; W[w11][col] += 1
        col += 1
        
        # M4 = a11(b10-b00)
        U[u11][col] += 1
        V[v10][col] += 1; V[v00][col] -= 1
        W[w00][col] += 1; W[w10][col] += 1
        col += 1
        
        # M5 = (a00+a01)b11
        U[u00][col] += 1; U[u01][col] += 1
        V[v11][col] += 1
        W[w00][col] -= 1; W[w01][col] += 1
        col += 1
        
        # M6 = (a10-a00)(b00+b01)
        U[u10][col] += 1; U[u00][col] -= 1
        V[v00][col] += 1; V[v01][col] += 1
        W[w11][col] += 1
        col += 1
        
        # M7 = (a01-a11)(b10+b11)
        U[u01][col] += 1; U[u11][col] -= 1
        V[v10][col] += 1; V[v11][col] += 1
        W[w00][col] += 1
        col += 1

    def add_schoolbook_2x1(a_row_offset, a_col_offset, b_row_offset, b_col_offset, c_row_offset, c_col_offset):
        nonlocal col
        # A is 2x2, B is 2x1, C is 2x1
        
        u00 = a_row_offset * m + a_col_offset
        u01 = a_row_offset * m + a_col_offset + 1
        u10 = (a_row_offset + 1) * m + a_col_offset
        u11 = (a_row_offset + 1) * m + a_col_offset + 1
        
        v0 = b_row_offset * p + b_col_offset
        v1 = (b_row_offset + 1) * p + b_col_offset
        
        w0 = c_col_offset * n + c_row_offset
        w1 = c_col_offset * n + c_row_offset + 1
        
        # A00 B0 -> C0
        U[u00][col] = 1; V[v0][col] = 1; W[w0][col] = 1
        col += 1
        # A01 B1 -> C0
        U[u01][col] = 1; V[v1][col] = 1; W[w0][col] = 1
        col += 1
        # A10 B0 -> C1
        U[u10][col] = 1; V[v0][col] = 1; W[w1][col] = 1
        col += 1
        # A11 B1 -> C1
        U[u11][col] = 1; V[v1][col] = 1; W[w1][col] = 1
        col += 1

    # Part 1: C1 = A B1 (2x2)
    # B1 is cols 0,1.
    add_strassen(0, 0, 0, 0, 0, 0)
    add_strassen(0, 2, 2, 0, 0, 0)
    
    # Part 2: C2 = A B2 (2x3)
    # B2 is cols 2,3,4.
    # Split B2 into B21 (col 2) and B22 (cols 3,4).
    
    # C21 = A B21 (2x1)
    add_schoolbook_2x1(0, 0, 0, 2, 0, 2)
    add_schoolbook_2x1(0, 2, 2, 2, 0, 2)
    
    # C22 = A B22 (2x2)
    add_strassen(0, 0, 0, 3, 0, 3)
    add_strassen(0, 2, 2, 3, 0, 3)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
