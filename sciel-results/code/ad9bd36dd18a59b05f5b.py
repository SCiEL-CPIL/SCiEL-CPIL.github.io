"""
Matrix multiplication tensor decomposition for 4x4 * 4x5.
Uses the schoolbook algorithm with rank 80.
This is the standard baseline.
"""

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a dictionary containing factor matrices U, V, W for the
    matrix multiplication tensor of dimensions 4x4 * 4x5.
    
    The decomposition is exact and uses the schoolbook rank (n*m*p).
    U: (n*m, rank)
    V: (m*p, rank)
    W: (p*n, rank)
    
    The tensor T[i, j, k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize matrices with zeros
    # Using list of lists for simplicity and to avoid numpy import overhead if not needed,
    # though numpy is allowed. Lists are sufficient for small dimensions.
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Each term in the schoolbook product corresponds to one rank-1 tensor
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
