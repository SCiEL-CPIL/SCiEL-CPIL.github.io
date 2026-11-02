"""Rewritten program: Strassen-based decomposition for 3x3x8 matrix multiplication."""
DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 3, 8
    rank = 68
    U = [[0]*rank for _ in range(n*m)]   # 9 x 68
    V = [[0]*rank for _ in range(m*p)]   # 24 x 68
    W = [[0]*rank for _ in range(p*n)]   # 24 x 68
    
    col = 0
    
    # Part 1: A[0:2,0:2] * B[0:2,0:8] using Strassen (28 mults)
    # Group columns into 4 pairs: (0,1), (2,3), (4,5), (6,7)
    for group in range(4):
        k0 = group * 2
        k1 = group * 2 + 1
        
        # m1 = (A[0,0]+A[1,1])*(B[0,k0]+B[1,k1])
        # Contributes to: C[0,k0]+1, C[0,k1]+1, C[1,k0]+1, C[1,k1]-1
        U[0][col] = 1; U[4][col] = 1
        V[k0][col] = 1; V[8+k1][col] = 1
        W[3*k0][col] = 1; W[3*k1][col] = 1; W[3*k0+1][col] = 1; W[3*k1+1][col] = -1
        col += 1
        
        # m2 = (A[1,0]+A[1,1])*B[0,k0]
        # Contributes to: C[1,k0]+1, C[1,k1]-1
        U[3][col] = 1; U[4][col] = 1
        V[k0][col] = 1
        W[3*k0+1][col] = 1; W[3*k1+1][col] = -1
        col += 1
        
        # m3 = A[0,0]*(B[0,k1]-B[1,k1])
        # Contributes to: C[0,k1]+1, C[1,k1]+1
        U[0][col] = 1
        V[k1][col] = 1; V[8+k1][col] = -1
        W[3*k1][col] = 1; W[3*k1+1][col] = 1
        col += 1
        
        # m4 = A[1,1]*(B[1,k0]-B[0,k0])
        # Contributes to: C[0,k0]+1, C[1,k0]+1
        U[4][col] = 1
        V[8+k0][col] = 1; V[k0][col] = -1
        W[3*k0][col] = 1; W[3*k0+1][col] = 1
        col += 1
        
        # m5 = (A[0,0]+A[0,1])*B[1,k1]
        # Contributes to: C[0,k0]-1, C[0,k1]+1
        U[0][col] = 1; U[1][col] = 1
        V[8+k1][col] = 1
        W[3*k0][col] = -1; W[3*k1][col] = 1
        col += 1
        
        # m6 = (A[1,0]-A[0,0])*(B[0,k0]+B[0,k1])
        # Contributes to: C[1,k1]+1
        U[3][col] = 1; U[0][col] = -1
        V[k0][col] = 1; V[k1][col] = 1
        W[3*k1+1][col] = 1
        col += 1
        
        # m7 = (A[0,1]-A[1,1])*(B[1,k0]+B[1,k1])
        # Contributes to: C[0,k0]+1, C[1,k1]+1
        U[1][col] = 1; U[4][col] = -1
        V[8+k0][col] = 1; V[8+k1][col] = 1
        W[3*k0][col] = 1; W[3*k1+1][col] = 1
        col += 1
    
    # Part 2: A[0:2,2] * B[2,0:8] (schoolbook, 16 mults)
    for i in range(2):
        for k in range(8):
            U[i*3+2][col] = 1
            V[16+k][col] = 1
            W[k*3+i][col] = 1
            col += 1
    
    # Part 3: A[2,0:2] * B[0:2,0:8] (schoolbook, 16 mults)
    for j in range(2):
        for k in range(8):
            U[6+j][col] = 1
            V[j*8+k][col] = 1
            W[k*3+2][col] = 1
            col += 1
    
    # Part 4: A[2,2] * B[2,0:8] (schoolbook, 8 mults)
    for k in range(8):
        U[8][col] = 1
        V[16+k][col] = 1
        W[k*3+2][col] = 1
        col += 1
    
    assert col == rank, f"Expected {rank}, got {col}"
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
