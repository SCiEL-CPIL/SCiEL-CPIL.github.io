DIMENSIONS = [2, 2, 8]

def construct():
    n, m, p = DIMENSIONS
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # n*m = 4, m*p = 16, p*n = 16
    
    # Strassen-based decomposition: 4 blocks of 2x2, each with 7 terms = 28 rank
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for b in range(4):  # 4 blocks
        k0 = 2 * b      # first column of this block
        k1 = 2 * b + 1  # second column of this block
        
        # M1 = (A00 + A11) * (B[0,k0] + B[1,k1])
        # U: A[0,0]=U[0], A[1,1]=U[3]
        U[0][col] += 1
        U[3][col] += 1
        # V: B[0,k0]=V[0*8+k0], B[1,k1]=V[1*8+k1]
        V[0 * p + k0][col] += 1
        V[1 * p + k1][col] += 1
        # W: C[0,k0]+=M1, C[1,k1]+=M1
        # W[k*n+i, col]: C[0,k0] -> W[k0*2+0], C[1,k1] -> W[k1*2+1]
        W[k0 * n + 0][col] += 1
        W[k1 * n + 1][col] += 1
        col += 1
        
        # M2 = (A10 + A11) * B[0,k0]
        U[2][col] += 1  # A[1,0]
        U[3][col] += 1  # A[1,1]
        V[0 * p + k0][col] += 1
        # W: C[1,k0]+=M2, C[1,k1]-=M2
        W[k0 * n + 1][col] += 1
        W[k1 * n + 1][col] += -1
        col += 1
        
        # M3 = A00 * (B[0,k1] - B[1,k1])
        U[0][col] += 1  # A[0,0]
        V[0 * p + k1][col] += 1
        V[1 * p + k1][col] += -1
        # W: C[0,k1]+=M3, C[1,k1]+=M3
        W[k1 * n + 0][col] += 1
        W[k1 * n + 1][col] += 1
        col += 1
        
        # M4 = A11 * (B[1,k0] - B[0,k0])
        U[3][col] += 1  # A[1,1]
        V[1 * p + k0][col] += 1
        V[0 * p + k0][col] += -1
        # W: C[0,k0]+=M4, C[1,k0]+=M4
        W[k0 * n + 0][col] += 1
        W[k0 * n + 1][col] += 1
        col += 1
        
        # M5 = (A00 + A01) * B[1,k1]
        U[0][col] += 1  # A[0,0]
        U[1][col] += 1  # A[0,1]
        V[1 * p + k1][col] += 1
        # W: C[0,k0]-=M5, C[0,k1]+=M5
        W[k0 * n + 0][col] += -1
        W[k1 * n + 0][col] += 1
        col += 1
        
        # M6 = (A10 - A00) * (B[0,k0] + B[0,k1])
        U[2][col] += 1  # A[1,0]
        U[0][col] += -1  # A[0,0]
        V[0 * p + k0][col] += 1
        V[0 * p + k1][col] += 1
        # W: C[1,k1]+=M6
        W[k1 * n + 1][col] += 1
        col += 1
        
        # M7 = (A01 - A11) * (B[1,k0] + B[1,k1])
        U[1][col] += 1  # A[0,1]
        U[3][col] += -1  # A[1,1]
        V[1 * p + k0][col] += 1
        V[1 * p + k1][col] += 1
        # W: C[0,k0]+=M7
        W[k0 * n + 0][col] += 1
        col += 1
    
    return {"U": U, "V": V, "W": W}
