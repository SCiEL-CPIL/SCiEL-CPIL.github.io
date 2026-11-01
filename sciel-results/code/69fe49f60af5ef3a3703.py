DIMENSIONS = [2, 2, 8]

def construct():
    n, m, p = DIMENSIONS
    rank = 28  # 4 blocks * 7 Strassen terms
    
    U = [[0] * rank for _ in range(n * m)]  # (4, 28)
    V = [[0] * rank for _ in range(m * p)]  # (16, 28)
    W = [[0] * rank for _ in range(p * n)]  # (16, 28)
    
    r = 0
    for b in range(4):  # 4 blocks of 2x2
        k0 = 2 * b  # first column index of this block
        k1 = 2 * b + 1  # second column index
        
        # M1 = (A[0,0]+A[1,1]) * (B[0,k0]+B[1,k1])
        # U: A[0,0]=1, A[1,1]=1
        U[0][r] = 1
        U[3][r] = 1
        # V: B[0,k0]=1, B[1,k1]=1
        V[0 * p + k0][r] = 1
        V[1 * p + k1][r] = 1
        # W: C[0,k0]+=1, C[1,k1]+=1
        W[k0 * n + 0][r] = 1
        W[k1 * n + 1][r] = 1
        r += 1
        
        # M2 = (A[1,0]+A[1,1]) * B[0,k0]
        U[2][r] = 1
        U[3][r] = 1
        V[0 * p + k0][r] = 1
        # W: C[1,k0]+=1, C[1,k1]-=1
        W[k0 * n + 1][r] = 1
        W[k1 * n + 1][r] = -1
        r += 1
        
        # M3 = A[0,0] * (B[0,k1]-B[1,k1])
        U[0][r] = 1
        V[0 * p + k1][r] = 1
        V[1 * p + k1][r] = -1
        # W: C[0,k1]+=1, C[1,k1]+=1
        W[k1 * n + 0][r] = 1
        W[k1 * n + 1][r] = 1
        r += 1
        
        # M4 = A[1,1] * (B[1,k0]-B[0,k0])
        U[3][r] = 1
        V[1 * p + k0][r] = 1
        V[0 * p + k0][r] = -1
        # W: C[0,k0]+=1, C[1,k0]+=1
        W[k0 * n + 0][r] = 1
        W[k0 * n + 1][r] = 1
        r += 1
        
        # M5 = (A[0,0]+A[0,1]) * B[1,k1]
        U[0][r] = 1
        U[1][r] = 1
        V[1 * p + k1][r] = 1
        # W: C[0,k0]-=1, C[0,k1]+=1
        W[k0 * n + 0][r] = -1
        W[k1 * n + 0][r] = 1
        r += 1
        
        # M6 = (A[1,0]-A[0,0]) * (B[0,k0]+B[0,k1])
        U[2][r] = 1
        U[0][r] = -1
        V[0 * p + k0][r] = 1
        V[0 * p + k1][r] = 1
        # W: C[1,k1]+=1
        W[k1 * n + 1][r] = 1
        r += 1
        
        # M7 = (A[0,1]-A[1,1]) * (B[1,k0]+B[1,k1])
        U[1][r] = 1
        U[3][r] = -1
        V[1 * p + k0][r] = 1
        V[1 * p + k1][r] = 1
        # W: C[0,k0]+=1
        W[k0 * n + 0][r] = 1
        r += 1
    
    return {"U": U, "V": V, "W": W}
