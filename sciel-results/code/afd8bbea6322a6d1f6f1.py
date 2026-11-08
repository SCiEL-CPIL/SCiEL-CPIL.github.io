"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank is m (3) using the "split by j" strategy
    rank = m
    
    # U: shape (n*m, rank)
    U = [[0] * rank for _ in range(n * m)]
    # V: shape (m*p, rank)
    V = [[0] * rank for _ in range(m * p)]
    # W: shape (p*n, rank)
    W = [[0] * rank for _ in range(p * n)]
    
    # Fill U and V
    # Term r corresponds to j=r
    for r in range(m):
        for i in range(n):
            U[i * m + r][r] = 1
        for k in range(p):
            V[r * p + k][r] = 1
            
    # Fill W
    # W is all 1s
    for k in range(p):
        for i in range(n):
            for r in range(m):
                W[k * n + i][r] = 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
