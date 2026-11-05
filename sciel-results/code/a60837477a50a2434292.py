"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the rank-minimal tensor decomposition for matrix multiplication
    of dimensions (2, 3, 5).
    
    The minimal rank for M_{2,3,5} is 2*3*5 = 30.
    We use the schoolbook decomposition which achieves this rank.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize matrices with zeros
    # U shape: (n*m, rank)
    # V shape: (m*p, rank)
    # W shape: (p*n, rank)
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Indexing for U: row is i*m + j
                U[i * m + j][col] = 1.0
                # Indexing for V: row is j*p + k
                V[j * p + k][col] = 1.0
                # Indexing for W: row is k*n + i
                W[k * n + i][col] = 1.0
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
