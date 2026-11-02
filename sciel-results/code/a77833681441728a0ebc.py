"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns the schoolbook decomposition for 4x5 by 5x6 matrix multiplication.
    Rank is n*m*p = 120.
    
    U: (n*m, rank)
    V: (m*p, rank)
    W: (p*n, rank) with index k*n + i
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize U, V, W with zeros
    # Using lists of lists as per the interface requirement (integer/half-integer entries)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Index mapping for U: row is i*m + j
                U[i * m + j][column] = 1
                # Index mapping for V: row is j*p + k
                V[j * p + k][column] = 1
                # Index mapping for W: row is k*n + i (Trace convention W[k*n+i, r])
                W[k * n + i][column] = 1
                column += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
