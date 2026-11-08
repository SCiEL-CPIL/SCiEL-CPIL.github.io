import time

DIMENSIONS = [4, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    # Total rank calculated: 92
    rank = 92
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a term
    def add_term(a_indices, b_indices, c_indices, coeff):
        nonlocal col
        # a_indices: list of (i, j)
        # b_indices: list of (j, k)
        # c_indices: list of (i, k)
        # coeff: list of (u_coeff, v_coeff, w_coeff) for each multiplication
        pass # Placeholder
    
    # We will manually iterate through the 18 terms
    
    # Define Strassen 2x2x2
    def add_strassen(a_r, a_c, b_r, b_c, c_r, c_c):
        nonlocal col
        # a_r, a_c, b_r, b_c, c_r, c_c are lists of 2 indices
        # A is 2x2, B is 2x2, C is 2x2
        # M1 = (A00 + A11)(B00 + B11)
        # M2 = (A10 + A11)B00
        # M3 = A00(B01 - B11)
        # M4 = A11(B10 - B00)
        # M5 = (A00 + A01)B11
        # M6 = (A10 - A11)(B01 + B11)
        # M7 = (A01 - A10)(B00 + B10)
        
        # C00 = M1 - M4 + M5 - M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M3 + M2 - M6
        
        # Indices in local 2x2
        # A: (0,0)->a_r[0]*m+a_c[0], (0,1)->a_r[0]*m+a_c[1], (1,0)->a_r[1]*m+a_c[0], (1,1)->a_r[1]*m+a_c[1]
        # B: (0,0)->b_r[0]*p+b_c[0], (0,1)->b_r[0]*p+b_c[1], (1,0)->b_r[1]*p+b_c[0], (1,1)->b_r[1]*p+b_c[1]
        # C: (0,0)->c_r[0]*n+c_c[0], (0,1)->c_r[0]*n+c_c[1], (1,0)->c_r[1]*n+c_c[0], (1,1)->c_r[1]*n+c_c[1]
        
        a00 = a_r[0] * m + a_c[0]
        a01 = a_r[0] * m + a_c[1]
        a10 = a_r[1] * m + a_c[0]
        a11 = a_r[1] * m + a_c[1]
        
        b00 = b_r[0] * p + b_c[0]
        b01 = b_r[0] * p + b_c[1]
        b10 = b_r[1] * p + b_c[0]
        b11 = b_r[1] * p + b_c[1]
        
        c00 = c_r[0] * n + c_c[0]
        c01 = c_r[0] * n + c_c[1]
        c10 = c_r[1] * n + c_c[0]
        c11 = c_r[1] * n + c_c[1]
        
        # M1
        U[a00][col] += 1; U[a11][col] += 1
        V[b00][col] += 1; V[b11][col] += 1
        W[c00][col] += 1; W[c11][col] += 1
        col += 1
        
        # M2
        U[a10][col] += 1; U[a11][col] += 1
        V[b00][col] += 1
        W[c10][col] += 1
        col += 1
        
        # M3
        U[a00][col] += 1
        V[b01][col] += 1; V[b11][col] -= 1
        W[c01][col] += 1
        col += 1
        
        # M4
        U[a11][col] += 1
        V[b10][col] += 1; V[b00][col] -= 1
        W[c00][col] -= 1; W[c10][col] += 1
        col += 1
        
        # M5
        U[a00][col] += 1; U[a01][col] += 1
        V[b11][col] += 1
        W[c00][col] += 1; W[c01][col] += 1
        col += 1
        
        # M6
        U[a10][col] += 1; U[a11][col] -= 1
        V[b01][col] += 1; V[b11][col] += 1
        W[c11][col] -= 1
        col += 1
        
        # M7
        U[a01][col] += 1; U[a10][col] -= 1
        V[b00][col] += 1; V[b10][col] += 1
        W[c00][col] -= 1
        col += 1

    # Helper for 2x1x2
    def add_2x1x2(a_r, a_c, b_r, b_c, c_r, c_c):
        nonlocal col
        # A is 2x1, B is 1x2, C is 2x2
        # A: a_r[0], a_r[1] (rows), a_c[0] (col)
        # B: b_r[0] (row), b_c[0], b_c[1] (cols)
        # C: c_r[0], c_r[1] (rows), c_c[0], c_c[1] (cols)
        
        a0 = a_r[0] * m + a_c[0]
        a1 = a_r[1] * m + a_c[0]
        
        b0 = b_r[0] * p + b_c[0]
        b1 = b_r[0] * p + b_c[1]
        
        c00 = c_r[0] * n + c_c[0]
        c01 = c_r[0] * n + c_c[1]
        c10 = c_r[1] * n + c_c[0]
        c11 = c_r[1] * n + c_c[1]
        
        # C00 = A0 B0
        U[a0][col] += 1; V[b0][col] += 1; W[c00][col] += 1
        col += 1
        # C01 = A0 B1
        U[a0][col] += 1; V[b1][col] += 1; W[c01][col] += 1
        col += 1
        # C10 = A1 B0
        U[a1][col] += 1; V[b0][col] += 1; W[c10][col] += 1
        col += 1
        # C11 = A1 B1
        U[a1][col] += 1; V[b1][col] += 1; W[c11][col] += 1
        col += 1

    # Helper for 2x2x1
    def add_2x2x1(a_r, a_c, b_r, b_c, c_r, c_c):
        nonlocal col
        # A is 2x2, B is 2x1, C is 2x1
        # A: a_r[0], a_r[1], a_c[0], a_c[1]
        # B: b_r[0], b_r[1], b_c[0]
        # C: c_r[0], c_r[1], c_c[0]
        
        a00 = a_r[0] * m + a_c[0]
        a01 = a_r[0] * m + a_c[1]
        a10 = a_r[1] * m + a_c[0]
        a11 = a_r[1] * m + a_c[1]
        
        b0 = b_r[0] * p + b_c[0]
        b1 = b_r[1] * p + b_c[0]
        
        c0 = c_r[0] * n + c_c[0]
        c1 = c_r[1] * n + c_c[0]
        
        # C0 = A00 B0 + A01 B1
        U[a00][col] += 1; V[b0][col] += 1; W[c0][col] += 1
        col += 1
        U[a01][col] += 1; V[b1][col] += 1; W[c0][col] += 1
        col += 1
        # C1 = A10 B0 + A11 B1
        U[a10][col] += 1; V[b0][col] += 1; W[c1][col] += 1
        col += 1
        U[a11][col] += 1; V[b1][col] += 1; W[c1][col] += 1
        col += 1

    # Helper for 2x1x1
    def add_2x1x1(a_r, a_c, b_r, b_c, c_r, c_c):
        nonlocal col
        # A is 2x1, B is 1x1, C is 2x1
        a0 = a_r[0] * m + a_c[0]
        a1 = a_r[1] * m + a_c[0]
        b0 = b_r[0] * p + b_c[0]
        c0 = c_r[0] * n + c_c[0]
        c1 = c_r[1] * n + c_c[0]
        
        U[a0][col] += 1; V[b0][col] += 1; W[c0][col] += 1
        col += 1
        U[a1][col] += 1; V[b0][col] += 1; W[c1][col] += 1
        col += 1

    # Indices
    # A rows: 0,1 (G1), 2,3 (G2)
    # A cols: 0,1 (G1), 2,3 (G2), 4 (G3)
    # B rows: 0,1 (G1), 2,3 (G2), 4 (G3)
    # B cols: 0,1 (G1), 2,3 (G2), 4 (G3)
    # C rows: 0,1 (G1), 2,3 (G2)
    # C cols: 0,1 (G1), 2,3 (G2), 4 (G3)
    
    a_r1 = [0, 1]
    a_r2 = [2, 3]
    a_c1 = [0, 1]
    a_c2 = [2, 3]
    a_c3 = [4]
    
    b_r1 = [0, 1]
    b_r2 = [2, 3]
    b_r3 = [4]
    b_c1 = [0, 1]
    b_c2 = [2, 3]
    b_c3 = [4]
    
    c_r1 = [0, 1]
    c_r2 = [2, 3]
    c_c1 = [0, 1]
    c_c2 = [2, 3]
    c_c3 = [4]
    
    # Term 1: A11 B11 (2x2x2)
    add_strassen(a_r1, a_c1, b_r1, b_c1, c_r1, c_c1)
    # Term 2: A12 B21 (2x2x2)
    add_strassen(a_r1, a_c2, b_r2, b_c1, c_r1, c_c1)
    # Term 3: A13 B31 (2x1x2)
    add_2x1x2(a_r1, a_c3, b_r3, b_c1, c_r1, c_c1)
    # Term 4: A11 B12 (2x2x2)
    add_strassen(a_r1, a_c1, b_r1, b_c2, c_r1, c_c2)
    # Term 5: A12 B22 (2x2x2)
    add_strassen(a_r1, a_c2, b_r2, b_c2, c_r1, c_c2)
    # Term 6: A13 B32 (2x1x2)
    add_2x1x2(a_r1, a_c3, b_r3, b_c2, c_r1, c_c2)
    # Term 7: A11 B13 (2x2x1)
    add_2x2x1(a_r1, a_c1, b_r1, b_c3, c_r1, c_c3)
    # Term 8: A12 B23 (2x2x1)
    add_2x2x1(a_r1, a_c2, b_r2, b_c3, c_r1, c_c3)
    # Term 9: A13 B33 (2x1x1)
    add_2x1x1(a_r1, a_c3, b_r3, b_c3, c_r1, c_c3)
    
    # Term 10: A21 B11 (2x2x2)
    add_strassen(a_r2, a_c1, b_r1, b_c1, c_r2, c_c1)
    # Term 11: A22 B21 (2x2x2)
    add_strassen(a_r2, a_c2, b_r2, b_c1, c_r2, c_c1)
    # Term 12: A23 B31 (2x1x2)
    add_2x1x2(a_r2, a_c3, b_r3, b_c1, c_r2, c_c1)
    # Term 13: A21 B12 (2x2x2)
    add_strassen(a_r2, a_c1, b_r1, b_c2, c_r2, c_c2)
    # Term 14: A22 B22 (2x2x2)
    add_strassen(a_r2, a_c2, b_r2, b_c2, c_r2, c_c2)
    # Term 15: A23 B32 (2x1x2)
    add_2x1x2(a_r2, a_c3, b_r3, b_c2, c_r2, c_c2)
    # Term 16: A21 B13 (2x2x1)
    add_2x2x1(a_r2, a_c1, b_r1, b_c3, c_r2, c_c3)
    # Term 17: A22 B23 (2x2x1)
    add_2x2x1(a_r2, a_c2, b_r2, b_c3, c_r2, c_c3)
    # Term 18: A23 B33 (2x1x1)
    add_2x1x1(a_r2, a_c3, b_r3, b_c3, c_r2, c_c3)
    
    return {"U": U, "V": V, "W": W}

