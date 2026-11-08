"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 3, 10
    
    num_blocks = p // 2  # 5
    rank_per_block = 11  # 7 (Strassen) + 4 (direct for j=2)
    total_rank = num_blocks * rank_per_block  # 55
    
    U = [[0.0] * total_rank for _ in range(n * m)]  # 6 rows
    V = [[0.0] * total_rank for _ in range(m * p)]  # 30 rows
    W = [[0.0] * total_rank for _ in range(p * n)]  # 20 rows
    
    col = 0
    for b in range(num_blocks):
        k0 = 2 * b
        k1 = 2 * b + 1
        
        # M1 = (A[0,0]+A[1,1]) * (B[0,k0]+B[1,k1])
        U[0][col] = 1.0; U[4][col] = 1.0
        V[0*10+k0][col] = 1.0; V[1*10+k1][col] = 1.0
        W[k0*2+0][col] = 1.0; W[k1*2+1][col] = 1.0
        col += 1
        
        # M2 = (A[1,0]+A[1,1]) * B[0,k0]
        U[3][col] = 1.0; U[4][col] = 1.0
        V[0*10+k0][col] = 1.0
        W[k0*2+1][col] = 1.0; W[k1*2+1][col] = -1.0
        col += 1
        
        # M3 = A[0,0] * (B[0,k1]-B[1,k1])
        U[0][col] = 1.0
        V[0*10+k1][col] = 1.0; V[1*10+k1][col] = -1.0
        W[k1*2+0][col] = 1.0; W[k1*2+1][col] = 1.0
        col += 1
        
        # M4 = A[1,1] * (B[1,k0]-B[0,k0])
        U[4][col] = 1.0
        V[1*10+k0][col] = 1.0; V[0*10+k0][col] = -1.0
        W[k0*2+0][col] = 1.0; W[k0*2+1][col] = 1.0
        col += 1
        
        # M5 = (A[0,0]+A[0,1]) * B[1,k1]
        U[0][col] = 1.0; U[1][col] = 1.0
        V[1*10+k1][col] = 1.0
        W[k0*2+0][col] = -1.0; W[k1*2+0][col] = 1.0
        col += 1
        
        # M6 = (A[1,0]-A[0,0]) * (B[0,k0]+B[0,k1])
        U[3][col] = 1.0; U[0][col] = -1.0
        V[0*10+k0][col] = 1.0; V[0*10+k1][col] = 1.0
        W[k1*2+1][col] = 1.0
        col += 1
        
        # M7 = (A[0,1]-A[1,1]) * (B[1,k0]+B[1,k1])
        U[1][col] = 1.0; U[4][col] = -1.0
        V[1*10+k0][col] = 1.0; V[1*10+k1][col] = 1.0
        W[k0*2+0][col] = 1.0
        col += 1
        
        # Direct terms for j=2:
        # A[0,2]*B[2,k0] -> C[0,k0]
        U[2][col] = 1.0; V[2*10+k0][col] = 1.0; W[k0*2+0][col] = 1.0
        col += 1
        # A[0,2]*B[2,k1] -> C[0,k1]
        U[2][col] = 1.0; V[2*10+k1][col] = 1.0; W[k1*2+0][col] = 1.0
        col += 1
        # A[1,2]*B[2,k0] -> C[1,k0]
        U[5][col] = 1.0; V[2*10+k0][col] = 1.0; W[k0*2+1][col] = 1.0
        col += 1
        # A[1,2]*B[2,k1] -> C[1,k1]
        U[5][col] = 1.0; V[2*10+k1][col] = 1.0; W[k1*2+1][col] = 1.0
        col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
