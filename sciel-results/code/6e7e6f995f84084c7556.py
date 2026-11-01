DIMENSIONS = [2, 2, 6]

def construct():
    n, m, p = DIMENSIONS
    rank = 21
    U = [[0] * rank for _ in range(n * m)]  # 4 x 21
    V = [[0] * rank for _ in range(m * p)]  # 12 x 21
    W = [[0] * rank for _ in range(p * n)]  # 12 x 21
    
    col = 0
    for g in range(3):
        k0 = 2 * g
        k1 = 2 * g + 1
        
        # Term 1: (A00+A11)(B[0,k0]+B[1,k1])
        U[0][col] = 1; U[3][col] = 1
        V[k0][col] = 1; V[p+k1][col] = 1
        W[k0*n+0][col] = 1; W[k1*n+1][col] = 1
        col += 1
        
        # Term 2: (A10+A11)B[0,k0]
        U[2][col] = 1; U[3][col] = 1
        V[k0][col] = 1
        W[k0*n+1][col] = 1
        col += 1
        
        # Term 3: A00(B[1,k0]-B[1,k1])
        U[0][col] = 1
        V[p+k0][col] = 1; V[p+k1][col] = -1
        W[k1*n+0][col] = 1; W[k1*n+1][col] = 1
        col += 1
        
        # Term 4: A11(B[0,k1]-B[0,k0])
        U[3][col] = 1
        V[k1][col] = 1; V[k0][col] = -1
        W[k0*n+0][col] = 1; W[k0*n+1][col] = 1
        col += 1
        
        # Term 5: (A00+A01)B[1,k1]
        U[0][col] = 1; U[1][col] = 1
        V[p+k1][col] = 1
        W[k0*n+0][col] = -1; W[k1*n+0][col] = 1
        col += 1
        
        # Term 6: (A10-A00)(B[0,k0]+B[1,k0])
        U[2][col] = 1; U[0][col] = -1
        V[k0][col] = 1; V[p+k0][col] = 1
        W[k1*n+1][col] = 1
        col += 1
        
        # Term 7: (A01-A11)(B[0,k1]+B[1,k1])
        U[1][col] = 1; U[3][col] = -1
        V[k1][col] = 1; V[p+k1][col] = 1
        W[k0*n+0][col] = 1; W[k1*n+1][col] = 1
        col += 1
    
    return {"U": U, "V": V, "W": W}
