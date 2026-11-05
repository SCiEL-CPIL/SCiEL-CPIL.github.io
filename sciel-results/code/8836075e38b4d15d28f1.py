"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We decompose into:
    # C = A * B
    # A is 2x4, B is 4x5.
    # Split B into B_L (4x2) and B_R (4x3).
    # C_L = A * B_L (2x2)
    # C_R = A * B_R (2x3)
    
    # C_L is 2x4 * 4x2.
    # Split A into A1, A2 (2x2) and B_L into B11, B21 (2x2).
    # C_L = A1*B11 + A2*B21.
    # Use Strassen (rank 7) for each 2x2x2 product. Total 14.
    
    # C_R is 2x4 * 4x3.
    # Split A into A1, A2 (2x2) and B_R into B12, B22 (2x3).
    # C_R = A1*B12 + A2*B22.
    # Use Schoolbook (rank 12) for each 2x2x3 product. Total 24.
    # Total rank = 14 + 24 = 38.
    
    rank = 38
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a 2x2x2 Strassen block
    def add_strassen_222(a_rows, b_rows, c_rows, c_cols):
        # a_rows: indices for A (2 rows)
        # b_rows: indices for B (2 rows)
        # c_rows: indices for C (2 rows)
        # c_cols: indices for C (2 cols)
        nonlocal col
        # A = [[a, b], [c, d]]
        # B = [[e, f], [g, h]]
        # Strassen products:
        # P1 = (a+d)(e+h)
        # P2 = (c+d)e
        # P3 = a(f-h)
        # P4 = d(g-f)
        # P5 = (a+c)(e+g)
        # P6 = (b+d)(h+g)
        # P7 = (b-c)(e+g)
        
        # Indices
        a_idx = a_rows[0]
        b_idx = a_rows[1]
        c_idx = a_rows[0] + 2 # Assuming A is 2x4, rows are 0,1. Cols 0,1 for A1, 2,3 for A2
        # Wait, A is 2x4.
        # A1 is cols 0,1. A2 is cols 2,3.
        # Rows are 0,1.
        
        # Let's define indices explicitly for clarity.
        # A1: U[0, col], U[1, col], U[2, col], U[3, col] ? No.
        # U shape (n*m, rank) = (8, rank).
        # A indices: i*m + j. i in 0..1, j in 0..3.
        # A1: i in 0..1, j in 0..1. Indices: 0, 1, 2, 3.
        # A2: i in 0..1, j in 2..3. Indices: 4, 5, 6, 7.
        
        # B indices: j*p + k. j in 0..3, k in 0..4.
        # B11: j in 0..1, k in 0..1. Indices: 0, 1, 4, 5.
        # B21: j in 2..3, k in 0..1. Indices: 8, 9, 12, 13.
        # B12: j in 0..1, k in 2..4. Indices: 2, 3, 6, 7, 10, 11.
        # B22: j in 2..3, k in 2..4. Indices: 14, 15, 18, 19, 22, 23.
        
        # C indices: k*n + i. k in 0..4, i in 0..1.
        # C_L: k in 0..1, i in 0..1. Indices: 0, 1, 2, 3.
        # C_R: k in 2..4, i in 0..1. Indices: 4, 5, 6, 7, 8, 9.
        
        # Strassen for A1 * B11 -> C_L
        # A1 elements: A[0,0]=0, A[0,1]=1, A[1,0]=2, A[1,1]=3.
        # B11 elements: B[0,0]=0, B[0,1]=1, B[1,0]=4, B[1,1]=5.
        # C_L elements: C[0,0]=0, C[0,1]=1, C[1,0]=2, C[1,1]=3.
        
        a00, a01, a10, a11 = 0, 1, 2, 3
        b00, b01, b10, b11 = 0, 1, 4, 5
        c00, c01, c10, c11 = 0, 1, 2, 3
        
        # P1 = (a00+a11)(b00+b11)
        U[a00][col] = 1; U[a11][col] = 1
        V[b00][col] = 1; V[b11][col] = 1
        W[c00][col] = 1; W[c11][col] = 1
        W[c01][col] = -1; W[c10][col] = -1
        col += 1
        
        # P2 = (a10+a11)b00
        U[a10][col] = 1; U[a11][col] = 1
        V[b00][col] = 1
        W[c10][col] = 1; W[c11][col] = 1
        col += 1
        
        # P3 = a00(b01-b11)
        U[a00][col] = 1
        V[b01][col] = 1; V[b11][col] = -1
        W[c01][col] = 1; W[c11][col] = 1
        col += 1
        
        # P4 = a11(b10-b01)
        U[a11][col] = 1
        V[b10][col] = 1; V[b01][col] = -1
        W[c00][col] = 1; W[c11][col] = 1
        col += 1
        
        # P5 = (a00+a10)(b00+b10)
        U[a00][col] = 1; U[a10][col] = 1
        V[b00][col] = 1; V[b10][col] = 1
        W[c00][col] = 1; W[c10][col] = 1
        W[c01][col] = -1; W[c11][col] = -1
        col += 1
        
        # P6 = (a01+a11)(b11+b10)
        U[a01][col] = 1; U[a11][col] = 1
        V[b11][col] = 1; V[b10][col] = 1
        W[c01][col] = 1; W[c11][col] = 1
        col += 1
        
        # P7 = (a01-a10)(b00+b10)
        U[a01][col] = 1; U[a10][col] = -1
        V[b00][col] = 1; V[b10][col] = 1
        W[c01][col] = 1; W[c10][col] = -1
        col += 1
        
        # Strassen for A2 * B21 -> C_L
        # A2 elements: A[0,2]=4, A[0,3]=5, A[1,2]=6, A[1,3]=7.
        # B21 elements: B[2,0]=8, B[2,1]=9, B[3,0]=12, B[3,1]=13.
        # C_L elements: Same as before 0,1,2,3.
        
        a00, a01, a10, a11 = 4, 5, 6, 7
        b00, b01, b10, b11 = 8, 9, 12, 13
        c00, c01, c10, c11 = 0, 1, 2, 3
        
        # P1
        U[a00][col] = 1; U[a11][col] = 1
        V[b00][col] = 1; V[b11][col] = 1
        W[c00][col] = 1; W[c11][col] = 1
        W[c01][col] = -1; W[c10][col] = -1
        col += 1
        
        # P2
        U[a10][col] = 1; U[a11][col] = 1
        V[b00][col] = 1
        W[c10][col] = 1; W[c11][col] = 1
        col += 1
        
        # P3
        U[a00][col] = 1
        V[b01][col] = 1; V[b11][col] = -1
        W[c01][col] = 1; W[c11][col] = 1
        col += 1
        
        # P4
        U[a11][col] = 1
        V[b10][col] = 1; V[b01][col] = -1
        W[c00][col] = 1; W[c11][col] = 1
        col += 1
        
        # P5
        U[a00][col] = 1; U[a10][col] = 1
        V[b00][col] = 1; V[b10][col] = 1
        W[c00][col] = 1; W[c10][col] = 1
        W[c01][col] = -1; W[c11][col] = -1
        col += 1
        
        # P6
        U[a01][col] = 1; U[a11][col] = 1
        V[b11][col] = 1; V[b10][col] = 1
        W[c01][col] = 1; W[c11][col] = 1
        col += 1
        
        # P7
        U[a01][col] = 1; U[a10][col] = -1
        V[b00][col] = 1; V[b10][col] = 1
        W[c01][col] = 1; W[c10][col] = -1
        col += 1
        
        # Schoolbook for A1 * B12 -> C_R
        # A1: 0,1,2,3
        # B12: 2,3,6,7,10,11 (j in 0,1; k in 2,3,4)
        # C_R: k in 2,3,4; i in 0,1.
        # C indices: k*n + i.
        # k=2: 4,5. k=3: 6,7. k=4: 8,9.
        
        a00, a01, a10, a11 = 0, 1, 2, 3
        b_idx = [2, 3, 6, 7, 10, 11] # b02, b03, b04, b12, b13, b14
        # b02=2, b03=3, b04=6
        # b12=7, b13=10, b14=11
        
        c02, c03, c04 = 4, 5, 6 # i=0, k=2,3,4
        c12, c13, c14 = 7, 8, 9 # i=1, k=2,3,4
        
        # 12 products
        # a00*b02
        U[a00][col] = 1; V[b_idx[0]][col] = 1
        W[c02][col] = 1
        col += 1
        # a00*b03
        U[a00][col] = 1; V[b_idx[1]][col] = 1
        W[c03][col] = 1
        col += 1
        # a00*b04
        U[a00][col] = 1; V[b_idx[2]][col] = 1
        W[c04][col] = 1
        col += 1
        # a01*b12
        U[a01][col] = 1; V[b_idx[3]][col] = 1
        W[c02][col] = 1
        col += 1
        # a01*b13
        U[a01][col] = 1; V[b_idx[4]][col] = 1
        W[c03][col] = 1
        col += 1
        # a01*b14
        U[a01][col] = 1; V[b_idx[5]][col] = 1
        W[c04][col] = 1
        col += 1
        # a10*b02
        U[a10][col] = 1; V[b_idx[0]][col] = 1
        W[c12][col] = 1
        col += 1
        # a10*b03
        U[a10][col] = 1; V[b_idx[1]][col] = 1
        W[c13][col] = 1
        col += 1
        # a10*b04
        U[a10][col] = 1; V[b_idx[2]][col] = 1
        W[c14][col] = 1
        col += 1
        # a11*b12
        U[a11][col] = 1; V[b_idx[3]][col] = 1
        W[c12][col] = 1
        col += 1
        # a11*b13
        U[a11][col] = 1; V[b_idx[4]][col] = 1
        W[c13][col] = 1
        col += 1
        # a11*b14
        U[a11][col] = 1; V[b_idx[5]][col] = 1
        W[c14][col] = 1
        col += 1
        
        # Schoolbook for A2 * B22 -> C_R
        # A2: 4,5,6,7
        # B22: 14,15,18,19,22,23
        # C_R: Same 4..9
        
        a00, a01, a10, a11 = 4, 5, 6, 7
        b_idx = [14, 15, 18, 19, 22, 23]
        
        U[a00][col] = 1; V[b_idx[0]][col] = 1
        W[c02][col] = 1
        col += 1
        U[a00][col] = 1; V[b_idx[1]][col] = 1
        W[c03][col] = 1
        col += 1
        U[a00][col] = 1; V[b_idx[2]][col] = 1
        W[c04][col] = 1
        col += 1
        U[a01][col] = 1; V[b_idx[3]][col] = 1
        W[c02][col] = 1
        col += 1
        U[a01][col] = 1; V[b_idx[4]][col] = 1
        W[c03][col] = 1
        col += 1
        U[a01][col] = 1; V[b_idx[5]][col] = 1
        W[c04][col] = 1
        col += 1
        U[a10][col] = 1; V[b_idx[0]][col] = 1
        W[c12][col] = 1
        col += 1
        U[a10][col] = 1; V[b_idx[1]][col] = 1
        W[c13][col] = 1
        col += 1
        U[a10][col] = 1; V[b_idx[2]][col] = 1
        W[c14][col] = 1
        col += 1
        U[a11][col] = 1; V[b_idx[3]][col] = 1
        W[c12][col] = 1
        col += 1
        U[a11][col] = 1; V[b_idx[4]][col] = 1
        W[c13][col] = 1
        col += 1
        U[a11][col] = 1; V[b_idx[5]][col] = 1
        W[c14][col] = 1
        col += 1
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
