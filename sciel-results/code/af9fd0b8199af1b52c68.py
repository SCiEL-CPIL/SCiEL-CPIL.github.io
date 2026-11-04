"""
Optimized program for matrix multiplication 2x3 * 3x9.
Uses a block-diagonal decomposition strategy where the computation for each row of the A matrix
is treated as an independent 1x3 * 3x9 multiplication.
"""
DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # The rank for a 1 x m by m x p multiplication is m * p.
    # Since the n rows of A are independent, we can decompose the problem
    # into n independent sub-problems of size 1 x m by m x p.
    rank_per_block = m * p
    total_rank = n * rank_per_block
    
    # Initialize factor matrices
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank) - Note: Index convention W[k*n + i, r]
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    # Iterate over each row i of A
    for i in range(n):
        # For this row i, we perform a 1 x m by m x p multiplication.
        # This requires m * p terms.
        for j in range(m):
            for k in range(p):
                # U index: i * m + j
                U[i * m + j][col] = 1
                
                # V index: j * p + k
                V[j * p + k][col] = 1
                
                # W index: k * n + i
                W[k * n + i][col] = 1
                
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
