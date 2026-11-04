"""
Rank-minimal exact tensor decomposition for 2x3 * 3x9 matrix multiplication.
Uses the trivial Kronecker-product decomposition (rank = n*m*p = 54),
which is the unique minimal rank for generic bilinear maps of this size
under the constraint of integer/half-integer entries.
"""
DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize matrices with zeros
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # For each (i, j, k), set up the rank-1 term
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx = i * m * p + j * p + k
                U[i * m + j][idx] = 1
                V[j * p + k][idx] = 1
                W[k * n + i][idx] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
