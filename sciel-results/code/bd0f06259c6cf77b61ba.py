"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 16]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=2, p=16
    num_blocks = p // 2  # 8 blocks of 2 columns each
    rank = num_blocks * 7  # 56
    
    U = [[0] * rank for _ in range(n * m)]  # (4, 56)
    V = [[0] * rank for _ in range(m * p)]  # (32, 56)
    W = [[0] * rank for _ in range(p * n)]  # (32, 56)
    
    for block in range(num_blocks):
        k = block * 2  # starting column of this block
        base = block * 7  # base rank index for this block
        
        # p1 = (A[0,0]+A[1,1]) * (B[0,k]+B[1,k+1])
        r = base + 0
        U[0][r] += 1   # A[0,0]
        U[3][r] += 1   # A[1,1]
        V[k][r] += 1       # B[0,k]
        V[16+k+1][r] += 1  # B[1,k+1]
        # C[0,k] += p1, C[1,k+1] += p1
        W[2*k][r] += 1       # C[0,k]
        W[2*(k+1)+1][r] += 1  # C[1,k+1]
        
        # p2 = (A[1,0]+A[1,1]) * B[0,k]
        r = base + 1
        U[2][r] += 1   # A[1,0]
        U[3][r] += 1   # A[1,1]
        V[k][r] += 1       # B[0,k]
        # C[1,k] += p2
        W[2*k+1][r] += 1  # C[1,k]
        
        # p3 = A[0,0] * (B[0,k+1]-B[1,k+1])
        r = base + 2
        U[0][r] += 1   # A[0,0]
        V[k+1][r] += 1       # B[0,k+1]
        V[16+k+1][r] -= 1  # B[1,k+1]
        # C[0,k+1] += p3, C[1,k+1] += p3
        W[2*(k+1)][r] += 1   # C[0,k+1]
        W[2*(k+1)+1][r] += 1  # C[1,k+1]
        
        # p4 = A[1,1] * (B[1,k]-B[0,k])
        r = base + 3
        U[3][r] += 1   # A[1,1]
        V[16+k][r] += 1   # B[1,k]
        V[k][r] -= 1       # B[0,k]
        # C[0,k] += p4, C[1,k] += p4
        W[2*k][r] += 1       # C[0,k]
        W[2*k+1][r] += 1  # C[1,k]
        
        # p5 = (A[0,0]+A[0,1]) * B[1,k+1]
        r = base + 4
        U[0][r] += 1   # A[0,0]
        U[1][r] += 1   # A[0,1]
        V[16+k+1][r] += 1  # B[1,k+1]
        # C[0,k] -= p5, C[0,k+1] += p5
        W[2*k][r] -= 1       # C[0,k]
        W[2*(k+1)][r] += 1   # C[0,k+1]
        
        # p6 = (A[1,0]-A[0,0]) * (B[0,k]+B[0,k+1])
        r = base + 5
        U[2][r] += 1   # A[1,0]
        U[0][r] -= 1   # A[0,0]
        V[k][r] += 1       # B[0,k]
        V[k+1][r] += 1       # B[0,k+1]
        # C[1,k+1] += p6
        W[2*(k+1)+1][r] += 1  # C[1,k+1]
        
        # p7 = (A[0,1]-A[1,1]) * (B[1,k]+B[1,k+1])
        r = base + 6
        U[1][r] += 1   # A[0,1]
        U[3][r] -= 1   # A[1,1]
        V[16+k][r] += 1   # B[1,k]
        V[16+k+1][r] += 1  # B[1,k+1]
        # C[0,k] += p7
        W[2*k][r] += 1       # C[0,k]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
