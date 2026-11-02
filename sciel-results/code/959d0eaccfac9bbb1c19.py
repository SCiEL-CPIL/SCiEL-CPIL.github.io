DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=3, m=3, p=8
    # Decompose A (3x3) into:
    # A11 (2x2), A12 (2x1), A21 (1x2), A22 (1x1)
    # B (3x8) into:
    # B1 (2x8), B2 (1x8)
    
    # Terms:
    # C11 (2x8) = A11 B1 + A12 B2
    # C21 (1x8) = A21 B1 + A22 B2
    
    # Ranks:
    # A11 B1: mu(2,2,8). Use 4 blocks of 2x2x2 (Strassen rank 7). Total 28.
    # A12 B2: mu(2,1,8). Schoolbook 2*1*8 = 16.
    # A21 B1: mu(1,2,8). Schoolbook 1*2*8 = 16.
    # A22 B2: mu(1,1,8). Schoolbook 1*1*8 = 8.
    # Total rank = 28 + 16 + 16 + 8 = 68.
    
    rank = 68
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # 1. A11 (rows 0-1, cols 0-1) * B1 (rows 0-1, cols 0-7)
    # Use 4 Strassen 2x2x2 blocks for the 8 columns.
    # Strassen for 2x2x2:
    # M1=(a11+a22)(b11+b22)
    # M2=(a21+a22)b11
    # M3=a11(b21-b22)
    # M4=a22(b12-b11)
    # M5=(a11+a12)b22
    # M6=(a21-a11)(b11+b12)
    # M7=(a12-a22)(b21+b22)
    # C11=M1+M4-M5+M7
    # C12=M3+M5
    # C21=M2+M4
    # C22=M1-M2+M3+M6
    
    for block in range(4):
        # Columns 2*block and 2*block+1 of B1
        # B1 rows are 0,1 of B.
        # A11 rows are 0,1 of A.
        # Output rows are 0,1 of C.
        
        # Map indices
        # a11 = U[0][col], a12 = U[1][col], a21 = U[3][col], a22 = U[4][col]
        # b11 = V[0][col], b12 = V[2][col], b21 = V[1][col], b22 = V[3][col]
        # Note: V index is j*p+k. j is row of B.
        # B1 is rows 0,1.
        # b11 is V[0*8 + 2*block]
        # b12 is V[0*8 + 2*block+1]
        # b21 is V[1*8 + 2*block]
        # b22 is V[1*8 + 2*block+1]
        
        # W index is k*n+i. k is col of C. i is row of C.
        # C11 is rows 0,1.
        # c11 is W[2*block*3 + 0]
        # c12 is W[2*block+1*3 + 0]
        # c21 is W[2*block*3 + 1]
        # c22 is W[2*block+1*3 + 1]
        
        # Strassen 1
        U[0][col] = 1; U[4][col] = 1
        V[0*8 + 2*block][col] = 1; V[1*8 + 2*block+1][col] = 1
        W[2*block*3 + 0][col] = 1
        W[2*block+1*3 + 1][col] = 1
        col += 1
        
        # Strassen 2
        U[3][col] = 1; U[4][col] = 1
        V[0*8 + 2*block][col] = 1
        W[2*block+1*3 + 1][col] = 1
        col += 1
        
        # Strassen 3
        U[0][col] = 1
        V[1*8 + 2*block][col] = 1; V[1*8 + 2*block+1][col] = -1
        W[2*block+1*3 + 0][col] = 1
        W[2*block+1*3 + 1][col] = 1
        col += 1
        
        # Strassen 4
        U[4][col] = 1
        V[0*8 + 2*block+1][col] = 1; V[0*8 + 2*block][col] = -1
        W[2*block*3 + 0][col] = 1
        W[2*block+1*3 + 1][col] = 1
        col += 1
        
        # Strassen 5
        U[0][col] = 1; U[1][col] = 1
        V[1*8 + 2*block+1][col] = 1
        W[2*block*3 + 0][col] = -1
        W[2*block+1*3 + 1][col] = -1
        col += 1
        
        # Strassen 6
        U[3][col] = 1; U[0][col] = -1
        V[0*8 + 2*block][col] = 1; V[0*8 + 2*block+1][col] = 1
        W[2*block*3 + 1][col] = 1
        W[2*block+1*3 + 1][col] = 1
        col += 1
        
        # Strassen 7
        U[1][col] = 1; U[4][col] = -1
        V[1*8 + 2*block][col] = 1; V[1*8 + 2*block+1][col] = 1
        W[2*block*3 + 0][col] = 1
        W[2*block+1*3 + 1][col] = -1
        col += 1

    # 2. A12 (rows 0-1, col 2) * B2 (row 2, cols 0-7)
    # Rank 16
    for r in range(2):
        for k in range(8):
            U[r*3 + 2][col] = 1
            V[2*8 + k][col] = 1
            W[k*3 + r][col] = 1
            col += 1

    # 3. A21 (row 2, cols 0-1) * B1 (rows 0-1, cols 0-7)
    # Rank 16
    for c in range(2):
        for k in range(8):
            U[2*3 + c][col] = 1
            V[c*8 + k][col] = 1
            W[k*3 + 2][col] = 1
            col += 1

    # 4. A22 (row 2, col 2) * B2 (row 2, cols 0-7)
    # Rank 8
    for k in range(8):
        U[2*3 + 2][col] = 1
        V[2*8 + k][col] = 1
        W[k*3 + 2][col] = 1
        col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
