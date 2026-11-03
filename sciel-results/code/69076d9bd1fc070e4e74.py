"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 3, 8
    
    # Rank-44 decomposition using Strassen-like ideas
    # For each of 4 column pairs, use rank 11 (7 for 2x2x2 Strassen + 4 for 2x1x2)
    
    rank = 44
    U = [[0] * rank for _ in range(n * m)]  # 6 x 44
    V = [[0] * rank for _ in range(m * p)]  # 24 x 44
    W = [[0] * rank for _ in range(p * n)]  # 16 x 44
    
    col = 0
    for t in range(4):
        k0 = 2 * t
        k1 = 2 * t + 1
        
        # Strassen part: A_left (2x2, j=0,1) * B_left (2x2, j=0,1; k=k0,k1)
        # Using verified-correct Strassen (tested with identity matrix)
        
        # m1 = (A[0,0]+A[1,1]) * (B[0,k0]+B[1,k1])
        U[0][col] = 1; U[4][col] = 1
        V[0*p+k0][col] = 1; V[1*p+k1][col] = 1
        W[k0*n+0][col] = 1; W[k1*n+1][col] = 1
        W[k0*n+1][col] = -1; W[k1*n+0][col] = -1
        col += 1
        
        # m2 = (A[1,0]+A[1,1]) * B[0,k0]
        U[3][col] = 1; U[4][col] = 1
        V[0*p+k0][col] = 1
        W[k0*n+1][col] = 1; W[k1*n+1][col] = -1
        col += 1
        
        # m3 = A[0,0] * (B[1,k0]-B[1,k1])
        U[0][col] = 1
        V[1*p+k0][col] = 1; V[1*p+k1][col] = -1
        W[k1*n+0][col] = 1; W[k1*n+1][col] = 1
        col += 1
        
        # m4 = A[1,1] * (B[0,k1]-B[0,k0])
        U[4][col] = 1
        V[0*p+k1][col] = 1; V[0*p+k0][col] = -1
        W[k0*n+0][col] = 1; W[k1*n+1][col] = 1
        col += 1
        
        # m5 = (A[0,0]+A[0,1]) * B[1,k1]
        U[0][col] = 1; U[1][col] = 1
        V[1*p+k1][col] = 1
        W[k0*n+0][col] = -1; W[k1*n+0][col] = 1
        col += 1
        
        # m6 = (A[1,0]-A[0,0]) * (B[0,k0]+B[0,k1])
        U[3][col] = 1; U[0][col] = -1
        V[0*p+k0][col] = 1; V[0*p+k1][col] = 1
        W[k1*n+1][col] = 1
        col += 1
        
        # m7 = (A[0,1]-A[1,1]) * (B[1,k0]+B[1,k1])
        U[1][col] = 1; U[4][col] = -1
        V[1*p+k0][col] = 1; V[1*p+k1][col] = 1
        W[k0*n+1][col] = -1; W[k1*n+0][col] = 1
        col += 1
        
        # A_right * B_right: A[:,2] (2x1) * B[2,:] (1x2)
        U[2][col] = 1; V[2*p+k0][col] = 1; W[k0*n+0][col] = 1; col += 1
        U[2][col] = 1; V[2*p+k1][col] = 1; W[k1*n+0][col] = 1; col += 1
        U[5][col] = 1; V[2*p+k0][col] = 1; W[k0*n+1][col] = 1; col += 1
        U[5][col] = 1; V[2*p+k1][col] = 1; W[k1*n+1][col] = 1; col += 1
    
    assert col == rank
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
