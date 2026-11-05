DIMENSIONS = [3, 5, 7]

def construct():
    n, m, p = DIMENSIONS  # n=3, m=5, p=7
    # Split m=5 into blocks of sizes 2, 2, 1
    # Block 0: A cols 0-1, B rows 0-1
    # Block 1: A cols 2-3, B rows 2-3
    # Block 2: A col 4, B row 4
    
    rank = 5
    U = [[0]*rank for _ in range(n*m)]  # 15 x 5
    V = [[0]*rank for _ in range(m*p)]  # 35 x 5
    W = [[0]*rank for _ in range(p*n)]  # 21 x 5
    
    # Block 0: A1 (3x2) * B1 (2x7)
    # Term 0: (A[i,0]+A[i,1]) * (B[0,k]+B[1,k]) * 1/2
    for i in range(n):
        U[i*m + 0][0] = 1
        U[i*m + 1][0] = 1
    for k in range(p):
        V[0*p + k][0] = 1
        V[1*p + k][0] = 1
    for i in range(n):
        for k in range(p):
            W[k*n + i][0] = 0.5
    
    # Term 1: (A[i,0]-A[i,1]) * (B[0,k]-B[1,k]) * 1/2
    for i in range(n):
        U[i*m + 0][1] = 1
        U[i*m + 1][1] = -1
    for k in range(p):
        V[0*p + k][1] = 1
        V[1*p + k][1] = -1
    for i in range(n):
        for k in range(p):
            W[k*n + i][1] = 0.5
    
    # Block 1: A2 (3x2) * B2 (2x7)
    # Term 2: (A[i,2]+A[i,3]) * (B[2,k]+B[3,k]) * 1/2
    for i in range(n):
        U[i*m + 2][2] = 1
        U[i*m + 3][2] = 1
    for k in range(p):
        V[2*p + k][2] = 1
        V[3*p + k][2] = 1
    for i in range(n):
        for k in range(p):
            W[k*n + i][2] = 0.5
    
    # Term 3: (A[i,2]-A[i,3]) * (B[2,k]-B[3,k]) * 1/2
    for i in range(n):
        U[i*m + 2][3] = 1
        U[i*m + 3][3] = -1
    for k in range(p):
        V[2*p + k][3] = 1
        V[3*p + k][3] = -1
    for i in range(n):
        for k in range(p):
            W[k*n + i][3] = 0.5
    
    # Block 2: A3 (3x1) * B3 (1x7)
    # Term 4: A[i,4] * B[4,k] * 1
    for i in range(n):
        U[i*m + 4][4] = 1
    for k in range(p):
        V[4*p + k][4] = 1
    for i in range(n):
        for k in range(p):
            W[k*n + i][4] = 1
    
    return {"U": U, "V": V, "W": W}
