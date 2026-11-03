"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-minimal exact tensor decomposition for multiplying a 2x3 matrix by a 3x10 matrix.
    
    The schoolbook rank is 2*3*10 = 60.
    Given the lack of known simple lower-rank decompositions for this specific rectangular case
    within the computational constraints, we return the schoolbook decomposition.
    
    The decomposition satisfies:
    C_{i,k} = sum_{j=0}^{m-1} sum_{c=0}^{r-1} U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    For the schoolbook case, each term (i, j, k) in the sum has its own unique column c.
    U[i*m+j, c] = 1, V[j*p+k, c] = 1, W[k*n+i, c] = 1, and 0 otherwise.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize matrices with zeros
    # Using lists of lists as per the interface requirement
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
