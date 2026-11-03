"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank is max(n*m, m*p, p*n) which is 30 for 2x3x10
    rank = m * p  # 30
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for c in range(rank):
        k = c // m  # 0 to p-1
        j = c % m   # 0 to m-1
        
        # V[j*p + k, c] = 1
        V[j * p + k][c] = 1
        
        # U[i*m + j, c] = 1 for all i in 0..n-1
        for i in range(n):
            U[i * m + j][c] = 1
            
        # W[k*n + i, c] = 1 for all i in 0..n-1
        for i in range(n):
            W[k * n + i][c] = 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
