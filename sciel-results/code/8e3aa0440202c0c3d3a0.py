"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    """Construct the rank-minimal (known) decomposition for 2x4 times 4x6.
    
    Using schoolbook: rank = 2*4*6 = 48.
    U: (n*m, rank) = (8, 48)
    V: (m*p, rank) = (24, 48)
    W: (p*n, rank) = (12, 48)
    
    Indexing: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    where c enumerates (i,j,k) triples.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p  # 48
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
