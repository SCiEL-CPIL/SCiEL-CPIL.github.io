"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We split the product into A * B_1 (3x3x3) and A * B_2 (3x3x2)
    # B_1 is first 3 columns, B_2 is last 2 columns.
    
    rank_total = 44
    U = [[0] * rank_total for _ in range(n * m)]
    V = [[0] * rank_total for _ in range(m * p)]
    W = [[0] * rank_total for _ in range(p * n)]
    
    col = 0
    
    # Part 1: 3x3x3 decomposition with rank 26
    # A = [[A22, a], [bT, c]], B1 = [[B22, b1], [b2T, d]]
    # A22 is 2x2, a is 2x1, b is 1x2, c is scalar.
    # B22 is 2x2, b1 is 2x1, b2 is 1x2, d is scalar.
    
    # Strassen for A22 * B22 (7 terms)
    # A22 = [[a00, a01], [a10, a11]]
    # B22 = [[b00, b01], [b10, b11]]
    # Strassen terms for 2x2x2:
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b01-b11)
    # M4 = a11(b10-b00)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b01)
    # M7 = (a01-a11)(b10+b11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # We need to map these to the full 3x3x3 tensor.
    # The indices for A22 are (0,0), (0,1), (1,0), (1,1).
    # The indices for B22 are (0,0), (0,1), (1,0), (1,1).
    # The output indices are (0,0), (0,1), (1,0), (1,1).
    
    # Strassen M1
    # U: a00, a11 -> 1
    # V: b00, b11 -> 1
    # W: c00, c11 -> 1 (with signs later)
    # Actually, we just add the terms.
    
    # Term 1: M1
    U[0][col] = 1; U[3][col] = 1 # a00, a11 (indices 0, 3 in 0-8? 0:0,0 1:0,1 2:1,0 3:1,1)
    V[0][col] = 1; V[4][col] = 1 # b00, b11 (0:0,0 1:0,1 2:1,0 3:1,1 ... wait B is 3x5. B22 is cols 0,1. Rows 0,1.
    # B indices: row 0, col 0 is 0. Row 0, col 1 is 1. Row 1, col 0 is 3. Row 1, col 1 is 4.
    # W indices: c00 is 0. c11 is 4? No, W is p*n = 5*3=15.
    # W[k*n+i]. c00 is k=0, i=0 -> 0. c11 is k=1, i=1 -> 1*3+1=4.
    W[0][col] = 1; W[4][col] = 1
    col += 1
    
    # Term 2: M2
    U[2][col] = 1; U[3][col] = 1 # a10, a11
    V[0][col] = 1 # b00
    W[3][col] = 1 # c10 (k=1, i=1? No, c10 is row 1, col 0. k=0, i=1? No.
    # C = AB. C_{ik} = sum_j A_{ij} B_{jk}.
    # C_{10} is i=1, k=0. W index k*n+i = 0*3+1 = 1.
    # C_{11} is i=1, k=1. W index 1*3+1 = 4.
    # C_{00} is i=0, k=0. W index 0.
    # C_{01} is i=0, k=1. W index 3.
    # Strassen:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # So M2 contributes to C10 (W index 1).
    W[1][col] = 1
    col += 1
    
    # Term 3: M3
    U[0][col] = 1 # a00
    V[1][col] = 1; V[4][col] = -1 # b01, b11
    W[3][col] = 1 # C01
    col += 1
    
    # Term 4: M4
    U[3][col] = 1 # a11
    V[2][col] = 1; V[0][col] = -1 # b10, b00
    W[0][col] = 1; W[1][col] = 1 # C00, C10
    col += 1
    
    # Term 5: M5
    U[0][col] = 1; U[1][col] = 1 # a00, a01
    V[4][col] = 1 # b11
    W[0][col] = -1; W[3][col] = 1 # C00, C01
    col += 1
    
    # Term 6: M6
    U[2][col] = 1; U[0][col] = -1 # a10, a00
    V[0][col] = 1; V[1][col] = 1 # b00, b01
    W[4][col] = 1 # C11
    col += 1
    
    # Term 7: M7
    U[1][col] = 1; U[3][col] = -1 # a01, a11
    V[2][col] = 1; V[3][col] = 1 # b10, b11? No, b10 is index 2, b11 is index 4?
    # B22 indices: 0(0,0), 1(0,1), 2(1,0), 3(1,1)? No.
    # B is 3x5.
    # B_{jk}. j=0,1 for B22. k=0,1 for B22.
    # j=0, k=0 -> 0
    # j=0, k=1 -> 1
    # j=1, k=0 -> 3
    # j=1, k=1 -> 4
    # So b10 is index 3, b11 is index 4.
    # In M7, we used b10, b11. So indices 3, 4.
    V[3][col] = 1; V[4][col] = 1
    W[0][col] = 1 # C00
    col += 1
    
    # Now the cross terms for 3x3x3.
    # a b2T: a is (2,0), (2,1) in A? No, a is row 0,1 col 2?
    # A = [[A22, a], [bT, c]].
    # A22 is rows 0,1 cols 0,1.
    # a is rows 0,1 col 2.
    # b is row 2 cols 0,1.
    # c is row 2 col 2.
    # B1 = [[B22, b1], [b2T, d]].
    # B22 is rows 0,1 cols 0,1.
    # b1 is rows 0,1 col 2.
    # b2 is row 2 cols 0,1.
    # d is row 2 col 2.
    
    # Terms: a_i b2_j (i=0,1; j=0,1)
    # U: a_i (index i*3+2). V: b2_j (index 2*5+j = 10+j). W: C_{ij} (index j*3+i).
    # i=0, j=0: U[2], V[10], W[0]
    # i=0, j=1: U[2], V[11], W[3]
    # i=1, j=0: U[5], V[10], W[1]
    # i=1, j=1: U[5], V[11], W[4]
    
    for i in range(2):
        for j in range(2):
            U[i*3+2][col] = 1
            V[10+j][col] = 1
            W[j*3+i][col] = 1
            col += 1
            
    # Terms: A22 b1 (2x2 * 2x1)
    # A22_{ij} b1_k (i,j=0,1; k=2? No, b1 is col 2 of B. So k=2.)
    # U: A22_{ij} (index i*3+j). V: b1 (index 2*5+? No, b1 is rows 0,1 col 2.
    # V index for b1_0 is 0*5+2=2. b1_1 is 1*5+2=7.
    # W: C_{i2} (index 2*3+i).
    
    for i in range(2):
        for j in range(2):
            U[i*3+j][col] = 1
            V[j*5+2][col] = 1
            W[2*3+i][col] = 1
            col += 1
            
    # Terms: a d (2x1 * scalar)
    # a_i d.
    # U: a_i (index i*3+2). V: d (index 2*5+2=12). W: C_{i2} (index 2*3+i).
    
    for i in range(2):
        U[i*3+2][col] = 1
        V[12][col] = 1
        W[2*3+i][col] = 1
        col += 1
        
    # Terms: bT B22 (1x2 * 2x2)
    # b_i B22_{jk} (i=0,1; j,k=0,1)
    # U: b_i (index 2*3+i). V: B22_{jk} (index j*5+k). W: C_{2k} (index k*3+2).
    
    for i in range(2):
        for j in range(2):
            for k in range(2):
                U[2*3+i][col] = 1
                V[j*5+k][col] = 1
                W[k*3+2][col] = 1
                col += 1
                
    # Terms: c b2T (scalar * 1x2)
    # c b2_j.
    # U: c (index 2*3+2=8). V: b2_j (index 2*5+j=10+j). W: C_{2j} (index j*3+2).
    
    for j in range(2):
        U[8][col] = 1
        V[10+j][col] = 1
        W[j*3+2][col] = 1
        col += 1
        
    # Terms: bT b1 (1x2 * 2x1)
    # b_i b1_i.
    # U: b_i (index 2*3+i). V: b1_i (index i*5+2). W: C_{22} (index 2*3+2=8).
    
    for i in range(2):
        U[2*3+i][col] = 1
        V[i*5+2][col] = 1
        W[8][col] = 1
        col += 1
        
    # Term: c d
    U[8][col] = 1
    V[12][col] = 1
    W[8][col] = 1
    col += 1
    
    # Part 2: 3x3x2 decomposition with rank 18 (Schoolbook)
    # A * B2. B2 is cols 3,4 of B.
    # A is 3x3. B2 is 3x2.
    # U: A_{ij}. V: B2_{jk}. W: C_{ik}.
    # j=0,1,2. k=3,4.
    
    for i in range(3):
        for j in range(3):
            for k in range(2):
                U[i*3+j][col] = 1
                V[j*5+(3+k)][col] = 1
                W[(3+k)*3+i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
