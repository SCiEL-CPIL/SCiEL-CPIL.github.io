import numpy as np

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Total rank is 8 * (7 + 4) + 4 * 6 = 8 * 11 + 24 = 112
    rank = 112
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col_offset = 0
    
    # Helper to get Strassen D222
    def get_D222():
        # U: 4x7, V: 4x7, W: 4x7
        u = [[0]*7 for _ in range(4)]
        v = [[0]*7 for _ in range(4)]
        w = [[0]*7 for _ in range(4)]
        
        # M1: (a+d)(e+h)
        u[0][0] = 1; u[1][0] = 1
        v[0][0] = 1; v[1][0] = 1
        w[0][0] = 1; w[3][0] = 1 # C11, C22
        
        # M2: (c+d)e
        u[2][1] = 1; u[3][1] = 1
        v[0][1] = 1
        w[2][1] = 1; w[3][1] = -1 # C21, C22
        
        # M3: a(f-h)
        u[0][2] = 1
        v[1][2] = 1; v[3][2] = -1
        w[1][2] = 1; w[3][2] = 1 # C12, C22
        
        # M4: d(g-e)
        u[3][3] = 1
        v[2][3] = 1; v[0][3] = -1
        w[0][3] = 1; w[2][3] = 1 # C11, C21
        
        # M5: (a+b)i
        u[0][4] = 1; u[1][4] = 1
        v[1][4] = 1
        w[0][4] = -1; w[1][4] = 1 # C11, C12
        
        # M6: (c-b)(g+h)
        u[2][5] = 1; u[1][5] = -1
        v[2][5] = 1; v[3][5] = 1
        w[3][5] = 1 # C22
        
        # M7: (a+d)(i-j)
        u[0][6] = 1; u[1][6] = 1
        v[1][6] = 1; v[3][6] = -1
        w[0][6] = 1 # C11
        
        return u, v, w

    # Helper to get D221
    def get_D221():
        # U: 4x4, V: 2x4, W: 2x4
        u = [[0]*4 for _ in range(4)]
        v = [[0]*4 for _ in range(2)]
        w = [[0]*4 for _ in range(2)]
        
        # M1: a*e
        u[0][0] = 1
        v[0][0] = 1
        w[0][0] = 1
        
        # M2: b*f
        u[1][1] = 1
        v[1][1] = 1
        w[1][1] = 1
        
        # M3: c*e
        u[2][2] = 1
        v[0][2] = 1
        w[0][2] = 1
        
        # M4: d*f
        u[3][3] = 1
        v[1][3] = 1
        w[1][3] = 1
        
        return u, v, w

    # Helper to get D213
    def get_D213():
        # U: 2x6, V: 3x6, W: 3x2
        u = [[0]*6 for _ in range(2)]
        v = [[0]*6 for _ in range(3)]
        w = [[0]*6 for _ in range(3)]
        
        # M1: a*e
        u[0][0] = 1
        v[0][0] = 1
        w[0][0] = 1
        
        # M2: a*f
        u[0][1] = 1
        v[1][1] = 1
        w[1][1] = 1
        
        # M3: a*g
        u[0][2] = 1
        v[2][2] = 1
        w[2][2] = 1
        
        # M4: b*e
        u[1][3] = 1
        v[0][3] = 1
        w[0][3] = 1
        
        # M5: b*f
        u[1][4] = 1
        v[1][4] = 1
        w[1][4] = 1
        
        # M6: b*g
        u[1][5] = 1
        v[2][5] = 1
        w[2][5] = 1
        
        return u, v, w

    d222 = get_D222()
    d221 = get_D221()
    d213 = get_D213()
    
    # Process 8 terms of type (2x2) * (2x3)
    # Terms: (A11,B11), (A12,B21), (A11,B12), (A12,B22), (A21,B11), (A22,B21), (A21,B12), (A22,B22)
    # A blocks:
    # A11: rows 0-1, cols 0-1
    # A12: rows 0-1, cols 2-3
    # A21: rows 2-3, cols 0-1
    # A22: rows 2-3, cols 2-3
    # B blocks:
    # B11: rows 0-1, cols 0-2
    # B12: rows 0-1, cols 3-5
    # B21: rows 2-3, cols 0-2
    # B22: rows 2-3, cols 3-5
    # C blocks:
    # C11: rows 0-1, cols 0-2
    # C12: rows 0-1, cols 3-5
    # C21: rows 2-3, cols 0-2
    # C22: rows 2-3, cols 3-5
    
    terms_223 = [
        # (A_row_start, A_col_start, B_row_start, B_col_start, C_row_start, C_col_start)
        (0, 0, 0, 0, 0, 0), # A11 * B11 -> C11
        (0, 2, 2, 0, 0, 0), # A12 * B21 -> C11
        (0, 0, 0, 3, 0, 3), # A11 * B12 -> C12
        (0, 2, 2, 3, 0, 3), # A12 * B22 -> C12
        (2, 0, 0, 0, 2, 0), # A21 * B11 -> C21
        (2, 2, 2, 0, 2, 0), # A22 * B21 -> C21
        (2, 0, 0, 3, 2, 3), # A21 * B12 -> C22
        (2, 2, 2, 3, 2, 3), # A22 * B22 -> C22
    ]
    
    for ar, ac, br, bc, cr, cc in terms_223:
        # Part 1: D222 for first 2 cols of B
        u, v, w = d222
        for i in range(4):
            for k in range(7):
                U[ar*m + ac + i][col_offset + k] += u[i][k]
        for i in range(4):
            for k in range(7):
                V[br*p + bc + i][col_offset + k] += v[i][k]
        for i in range(4):
            for k in range(7):
                W[(cc)*n + cr + i][col_offset + k] += w[i][k]
        col_offset += 7
        
        # Part 2: D221 for last col of B
        u, v, w = d221
        for i in range(4):
            for k in range(4):
                U[ar*m + ac + i][col_offset + k] += u[i][k]
        # B col is bc + 2
        for i in range(2):
            for k in range(4):
                V[(br + i)*p + (bc + 2)][col_offset + k] += v[i][k]
        # C col is cc + 2
        for i in range(2):
            for k in range(4):
                W[(cc + 2)*n + (cr + i)][col_offset + k] += w[i][k]
        col_offset += 4

    # Process 4 terms of type (2x1) * (1x3)
    # Terms: (A13,B31), (A13,B32), (A23,B31), (A23,B32)
    # A blocks:
    # A13: rows 0-1, col 4
    # A23: rows 2-3, col 4
    # B blocks:
    # B31: row 4, cols 0-2
    # B32: row 4, cols 3-5
    # C blocks:
    # C11: rows 0-1, cols 0-2
    # C12: rows 0-1, cols 3-5
    # C21: rows 2-3, cols 0-2
    # C22: rows 2-3, cols 3-5
    
    terms_213 = [
        # (A_row_start, B_col_start, C_row_start, C_col_start)
        (0, 0, 0, 0), # A13 * B31 -> C11
        (0, 3, 0, 3), # A13 * B32 -> C12
        (2, 0, 2, 0), # A23 * B31 -> C21
        (2, 3, 2, 3), # A23 * B32 -> C22
    ]
    
    for ar, bc, cr, cc in terms_213:
        u, v, w = d213
        for i in range(2):
            for k in range(6):
                U[ar*m + 4][col_offset + k] += u[i][k]
        for i in range(3):
            for k in range(6):
                V[4*p + (bc + i)][col_offset + k] += v[i][k]
        for i in range(3):
            for k in range(6):
                W[(cc + i)*n + cr][col_offset + k] += w[i][k]
        col_offset += 6

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
