"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the schoolbook tensor decomposition for matrix multiplication
    of dimensions 3x4 and 4x6.
    
    The rank is n*m*p = 3*4*6 = 72.
    U shape: (n*m, rank) = (12, 72)
    V shape: (m*p, rank) = (24, 72)
    W shape: (p*n, rank) = (18, 72)
    
    The decomposition is:
    U[i*m + j, c] = 1 if c == (i*m + j)*p + k
    V[j*p + k, c] = 1 if c == (i*m + j)*p + k
    W[k*n + i, c] = 1 if c == (i*m + j)*p + k
    
    This ensures that sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    only when the indices match the specific product term, which corresponds
    to the standard matrix multiplication.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize matrices with zeros
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Populate the decomposition
    # Each column c corresponds to a unique (i, j, k) tuple
    # c = (i * m + j) * p + k
    for i in range(n):
        for j in range(m):
            for k in range(p):
                c = (i * m + j) * p + k
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
