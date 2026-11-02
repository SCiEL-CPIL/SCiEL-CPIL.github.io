DIMENSIONS = [2, 3, 8]

def construct():
    n, m, p = DIMENSIONS  # n=2, m=3, p=8
    rank = 44
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    col = 0
    
    # Part 1: A_right (2x1) * B_top (1x8) -> 16 schoolbook
    for i in range(n):
        for k in range(p):
            U[i*m + 2][col] = 1
            V[0*p + k][col] = 1
            W[k*n + i][col] = 1
            col += 1
    
    # Part 2: A_left (2x2) * B_bottom (2x8) -> 4 Strassen blocks
    for b in range(4):
        k0 = 2*b
        k1 = 2*b + 1
        
        # M1 = (A00+A11)(B00+B11)
        U[0][col] = 1; U[4][col] = 1  # A[0,0], A[1,1]
        V[8+k0][col] = 1; V[16+k1][col] = 1  # B[1,k0], B[2,k1]
        W[k0*n+0][col] = 1; W[k1*n+1][col] = -1  # C[0,k0]+, C[1,k1]-
        col += 1
        
        # M2 = (A10+A11)B00
        U[3][col] = 1; U[4][col] = 1  # A[1,0], A[1,1]
        V[8+k0][col] = 1  # B[1,k0]
        W[k0*n+1][col] = 1; W[k1*n+1][col] = -1  # C[1,k0]+, C[1,k1]-
        col += 1
        
        # M3 = A00(B10-B11)
        U[0][col] = 1  # A[0,0]
        V[8+k0][col] = 1; V[8+k1][col] = -1  # B[1,k0], B[1,k1]
        W[k1*n+0][col] = 1; W[k1*n+1][col] = 1  # C[0,k1]+, C[1,k1]+
        col += 1
        
        # M4 = A11(B01-B00)
        U[4][col] = 1  # A[1,1]
        V[8+k1][col] = 1; V[8+k0][col] = -1  # B[1,k1], B[1,k0]
        W[k0*n+0][col] = 1; W[k0*n+1][col] = 1  # C[0,k0]+, C[1,k0]+
        col += 1
        
        # M5 = (A00+A01)B11
        U[0][col] = 1; U[1][col] = 1  # A[0,0], A[0,1]
        V[16+k1][col] = 1  # B[2,k1]
        W[k0*n+0][col] = -1; W[k1*n+0][col] = 1  # C[0,k0]-, C[0,k1]+
        col += 1
        
        # M6 = (A10-A00)(B00+B10)
        U[3][col] = 1; U[0][col] = -1  # A[1,0], A[0,0]
        V[8+k0][col] = 1; V[16+k0][col] = 1  # B[1,k0], B[2,k0]
        W[k1*n+1][col] = 1  # C[1,k1]+
        col += 1
        
        # M7 = (A01-A11)(B01+B11)
        U[1][col] = 1; U[4][col] = -1  # A[0,1], A[1,1]
        V[8+k1][col] = 1; V[16+k1][col] = 1  # B[1,k1], B[2,k1]
        W[k0*n+0][col] = 1  # C[0,k0]+
        col += 1
    
    return {"U": U, "V": V, "W": W}
