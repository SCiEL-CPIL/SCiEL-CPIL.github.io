"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank m * p = 4 * 7 = 28
    rank = m * p
    
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Decomposition:
    # For each j in 0..m-1, we use p terms.
    # Total rank = m * p.
    # Term index r = j * p + k
    # U[i*m + j, r] = 1 if j matches, else 0.
    # V[j*p + k, r] = 1 if j matches and k matches, else 0.
    # W[k*n + i, r] = 1 if k matches, else 0.
    
    for j in range(m):
        for k in range(p):
            r = j * p + k
            # U
            for i in range(n):
                U[i * m + j][r] = 1
            # V
            V[j * p + k][r] = 1
            # W
            for i in range(n):
                W[k * n + i][r] = 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
