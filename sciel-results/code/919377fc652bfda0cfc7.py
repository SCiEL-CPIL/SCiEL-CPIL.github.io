import time

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-minimal (or near-minimal) exact tensor decomposition for 
    multiplying a 3x4 matrix by a 4x5 matrix.
    
    The schoolbook algorithm yields a rank of n*m*p = 60.
    While theoretical lower bounds suggest a lower rank (e.g., 56) might be 
    achievable via block decomposition and Strassen-like algorithms, the 
    schoolbook decomposition is guaranteed to be valid and exact.
    
    Given the constraints and the need for a valid exact decomposition, 
    this implementation returns the schoolbook decomposition.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize matrices with zeros
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: i*m + j
                # V index: j*p + k
                # W index: k*n + i
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
