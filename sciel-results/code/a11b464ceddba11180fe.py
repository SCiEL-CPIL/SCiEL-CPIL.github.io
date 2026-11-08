"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 27 decomposition for 3x3x6
    # Uses two 3x3x3 schoolbook decompositions with shared U
    rank = 27
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Generate schoolbook indices for 3x3x3
    # i in 0..2, j in 0..2, k in 0..2
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(3): # Only 3 k's for the 3x3x3 block
                # U index: i*m + j
                U[i * m + j][col] = 1
                # V index: j*p + k  (for first block k=0..2)
                V[j * p + k][col] = 1
                # V index: j*p + (k+3) (for second block k=3..5)
                V[j * p + k + 3][col] = 1
                # W index: k*n + i   (for first block)
                W[k * n + i][col] = 1
                # W index: (k+3)*n + i (for second block)
                W[(k + 3) * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
