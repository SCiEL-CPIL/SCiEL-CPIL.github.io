"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 3, 8
    rank = 44
    U = [[0] * rank for _ in range(n * m)]  # (6, 44)
    V = [[0] * rank for _ in range(m * p)]  # (24, 44)
    W = [[0] * rank for _ in range(p * n)]  # (16, 44)
    
    col = 0
    # Part 1: Strassen on A1 (2x2) * B1 (2x8), 4 column pairs, 7 terms each = 28
    for k_pair in range(4):  # column pairs (0,1), (2,3), (4,5), (6,7)
        k0 = k_pair * 2
        k1 = k_pair * 2 + 1
        
        # M1 = (A[0,0]+A[1,1]) * (B[0,k0]+B[1,k1])
        U[0*m+0][col] = 1
        U[1*m+1][col] = 1
        V[0*p+k0][col] = 1
        V[1*p+k1][col] = 1
        W[k0*n+0][col] = 1  # C[0,k0] += M1
        W[k1*n+1][col] = 1  # C[1,k1] += M1
        col += 1
        
        # M2 = (A[1,0]+A[1,1]) * B[0,k0]
        U[1*m+0][col] = 1
        U[1*m+1][col] = 1
        V[0*p+k0][col] = 1
        W[k0*n+1][col] = 1  # C[1,k0] += M2
        col += 1
        
        # M3 = A[0,0] * (B[1,k1]-B[0,k1])
        U[0*m+0][col] = 1
        V[1*p+k1][col] = 1
        V[0*p+k1][col] = -1
        W[k1*n+0][col] = 1  # C[0,k1] += M3
        W[k1*n+1][col] = 1  # C[1,k1] += M3
        col += 1
        
        # M4 = A[1,1] * (B[1,k0]-B[0,k0])
        U[1*m+1][col] = 1
        V[1*p+k0][col] = 1
        V[0*p+k0][col] = -1
        W[k0*n+0][col] = 1  # C[0,k0] += M4
        W[k0*n+1][col] = 1  # C[1,k0] += M4
        col += 1
        
        # M5 = (A[0,0]+A[0,1]) * B[1,k1]
        U[0*m+0][col] = 1
        U[0*m+1][col] = 1
        V[1*p+k1][col] = 1
        W[k0*n+0][col] = -1 # C[0,k0] -= M5
        W[k1*n+0][col] = 1  # C[0,k1] += M5
        col += 1
        
        # M6 = (A[1,0]-A[0,0]) * (B[0,k0]+B[1,k0])
        U[1*m+0][col] = 1
        U[0*m+0][col] = -1
        V[0*p+k0][col] = 1
        V[1*p+k0][col] = 1
        W[k1*n+1][col] = 1  # C[1,k1] += M6
        col += 1
        
        # M7 = (A[0,1]-A[1,1]) * (B[1,k0]+B[1,k1])
        U[0*m+1][col] = 1
        U[1*m+1][col] = -1
        V[1*p+k0][col] = 1
        V[1*p+k1][col] = 1
        W[k0*n+0][col] = 1  # C[0,k0] += M7
        col += 1
    
    # Part 2: Schoolbook on A2 (2x1) * B2 (1x8), 16 terms
    for i in range(2):
        for k in range(8):
            U[i*m+2][col] = 1
            V[2*p+k][col] = 1
            W[k*n+i][col] = 1
            col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
