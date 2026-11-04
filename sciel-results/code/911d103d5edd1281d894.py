import time

def construct():
    n, m, p = 3, 3, 4
    rank = 34
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add Strassen terms for 2x2x2
    # A rows a_r1, a_r2; cols a_c1, a_c2
    # B rows b_r1, b_r2; cols b_c1, b_c2
    # C rows c_r1, c_r2; cols c_c1, c_c2
    def add_strassen(a_r1, a_r2, a_c1, a_c2, b_r1, b_r2, b_c1, b_c2, c_r1, c_r2, c_c1, c_c2):
        nonlocal col
        # M1 = (A11+A22)(B11+B22)
        # A terms: (A11, A22)
        # B terms: (B11, B22)
        # C terms: (C11, C22)
        # Coeffs:
        # M1: C11 += 1, C22 += 1
        # M2: C21 += 1
        # M3: C12 += 1
        # M4: C11 += 1, C21 += 1
        # M5: C11 -= 1, C12 += 1
        # M6: C22 += 1
        # M7: C11 += 1
        
        # Indices in local 2x2 grid:
        # A: 00, 01, 10, 11
        # B: 00, 01, 10, 11
        # C: 00, 01, 10, 11
        
        # M1: (A00+A11)(B00+B11)
        # U: A00=1, A11=1
        # V: B00=1, B11=1
        # W: C00=1, C11=1
        U[a_r1*m + a_c1][col] = 1
        U[a_r2*m + a_c2][col] = 1
        V[b_r1*p + b_c1][col] = 1
        V[b_r2*p + b_c2][col] = 1
        W[c_c1*n + c_r1][col] = 1
        W[c_c2*n + c_r2][col] = 1
        col += 1
        
        # M2: (A10+A11)B00
        # U: A10=1, A11=1
        # V: B00=1
        # W: C10=1
        U[a_r2*m + a_c1][col] = 1
        U[a_r2*m + a_c2][col] = 1
        V[b_r1*p + b_c1][col] = 1
        W[c_c1*n + c_r2][col] = 1
        col += 1
        
        # M3: A00(B01-B11)
        # U: A00=1
        # V: B01=1, B11=-1
        # W: C01=1
        U[a_r1*m + a_c1][col] = 1
        V[b_r1*p + b_c2][col] = 1
        V[b_r2*p + b_c2][col] = -1
        W[c_c2*n + c_r1][col] = 1
        col += 1
        
        # M4: A11(B10-B00)
        # U: A11=1
        # V: B10=1, B00=-1
        # W: C00=1, C10=1
        U[a_r2*m + a_c2][col] = 1
        V[b_r2*p + b_c1][col] = 1
        V[b_r1*p + b_c1][col] = -1
        W[c_c1*n + c_r1][col] = 1
        W[c_c1*n + c_r2][col] = 1
        col += 1
        
        # M5: (A00+A01)B11
        # U: A00=1, A01=1
        # V: B11=1
        # W: C00=-1, C01=1
        U[a_r1*m + a_c1][col] = 1
        U[a_r1*m + a_c2][col] = 1
        V[b_r2*p + b_c2][col] = 1
        W[c_c1*n + c_r1][col] = -1
        W[c_c2*n + c_r1][col] = 1
        col += 1
        
        # M6: (A10-A00)(B00+B01)
        # U: A10=1, A00=-1
        # V: B00=1, B01=1
        # W: C11=1
        U[a_r2*m + a_c1][col] = 1
        U[a_r1*m + a_c1][col] = -1
        V[b_r1*p + b_c1][col] = 1
        V[b_r1*p + b_c2][col] = 1
        W[c_c2*n + c_r2][col] = 1
        col += 1
        
        # M7: (A01-A11)(B10+B11)
        # U: A01=1, A11=-1
        # V: B10=1, B11=1
        # W: C00=1
        U[a_r1*m + a_c2][col] = 1
        U[a_r2*m + a_c2][col] = -1
        V[b_r2*p + b_c1][col] = 1
        V[b_r2*p + b_c2][col] = 1
        W[c_c1*n + c_r1][col] = 1
        col += 1

    # 1. A11 B11 (2x2x2)
    # A: rows 0,1; cols 0,1
    # B: rows 0,1; cols 0,1
    # C: rows 0,1; cols 0,1
    add_strassen(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
    
    # 2. A11 B12 (2x2x2)
    # A: rows 0,1; cols 0,1
    # B: rows 0,1; cols 2,3
    # C: rows 0,1; cols 2,3
    add_strassen(0, 1, 0, 1, 0, 1, 2, 3, 0, 1, 2, 3)
    
    # 3. A12 B21 (2x1x2)
    # A: rows 0,1; col 2
    # B: row 2; cols 0,1
    # C: rows 0,1; cols 0,1
    # Terms: A[i,2] B[2,k] C[i,k]
    for i in range(2):
        for k in range(2):
            U[i*m + 2][col] = 1
            V[2*p + k][col] = 1
            W[k*n + i][col] = 1
            col += 1
            
    # 4. A12 B22 (2x1x2)
    # A: rows 0,1; col 2
    # B: row 2; cols 2,3
    # C: rows 0,1; cols 2,3
    for i in range(2):
        for k in range(2, 4):
            U[i*m + 2][col] = 1
            V[2*p + k][col] = 1
            W[k*n + i][col] = 1
            col += 1
            
    # 5. A21 B11 (1x2x2)
    # A: row 2; cols 0,1
    # B: rows 0,1; cols 0,1
    # C: row 2; cols 0,1
    # Terms: A[2,j] B[j,k] C[2,k]
    for j in range(2):
        for k in range(2):
            U[2*m + j][col] = 1
            V[j*p + k][col] = 1
            W[k*n + 2][col] = 1
            col += 1
            
    # 6. A21 B12 (1x2x2)
    # A: row 2; cols 0,1
    # B: rows 0,1; cols 2,3
    # C: row 2; cols 2,3
    for j in range(2):
        for k in range(2, 4):
            U[2*m + j][col] = 1
            V[j*p + k][col] = 1
            W[k*n + 2][col] = 1
            col += 1
            
    # 7. A22 B21 (1x1x2)
    # A: row 2; col 2
    # B: row 2; cols 0,1
    # C: row 2; cols 0,1
    for k in range(2):
        U[2*m + 2][col] = 1
        V[2*p + k][col] = 1
        W[k*n + 2][col] = 1
        col += 1
        
    # 8. A22 B22 (1x1x2)
    # A: row 2; col 2
    # B: row 2; cols 2,3
    # C: row 2; cols 2,3
    for k in range(2, 4):
        U[2*m + 2][col] = 1
        V[2*p + k][col] = 1
        W[k*n + 2][col] = 1
        col += 1
        
    return {"U": U, "V": V, "W": W}
