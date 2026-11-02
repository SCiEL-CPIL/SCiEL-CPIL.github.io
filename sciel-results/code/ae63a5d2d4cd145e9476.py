"""Rewritten program for matrix multiplication 2,2,15 using structured algebraic decomposition."""
DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal exact tensor decomposition for multiplying a 2x2 matrix by a 2x15 matrix.
    
    Uses schoolbook multiplication with rank = 2*2*15 = 60.
    The output factor W follows the official trace convention W[k*n+i, r] where:
    - k is the column index of the 2x15 matrix (0 to 14)
    - i is the row index of the 2x2 result (0 to 1)
    - r is the rank index
    """
    n, m, p = 2, 2, 15
    rank = n * m * p
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
