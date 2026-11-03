import time

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns an exact tensor decomposition for multiplying a 4x5 matrix by a 5x5 matrix.
    
    Uses the schoolbook algorithm with rank n*m*p = 4*5*5 = 100.
    U: (n*m, rank)
    V: (m*p, rank)
    W: (p*n, rank)
    
    Entries are integers (0 or 1).
    """
    n, m, p = DIMENSIONS  # 4, 5, 5
    rank = n * m * p  # 100
    
    # Initialize matrices with zeros
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    # Iterate over all (i, j, k) combinations
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: i*m + j
                # V index: j*p + k
                # W index: k*n + i  (Convention: W[k*n+i, r])
                
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                
                column += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
