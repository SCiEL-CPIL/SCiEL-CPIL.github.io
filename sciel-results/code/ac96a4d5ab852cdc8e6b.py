"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank is n * p because for each output row i (n rows) and output col k (p cols),
    # we need p multiplications? No.
    # For 1 x m x p, rank is p.
    # For n x m x p, rank is n * p.
    # Here n=3, p=4, so rank = 12.
    rank = n * p
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for i in range(n):
        for k in range(p):
            r = i * p + k
            # U: index i*m + j
            for j in range(m):
                U[i * m + j][r] = 1
            # V: index j*p + k
            for j in range(m):
                V[j * p + k][r] = 1
            # W: index k*n + i
            W[k * n + i][r] = 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
