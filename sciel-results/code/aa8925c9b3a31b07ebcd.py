"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 8 decomposition
    # Split m=4 into 4 blocks of size 1.
    # Each block 2x1x5 has rank 2.
    rank = 8
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for j in range(m):
        # Block j uses columns col and col+1
        # U indices: i * m + j
        # V indices: j * p + k
        # W indices: k * n + i
        
        # r=0 (col)
        # i=0
        U[0 * m + j][col] = 1
        # i=1 is 0
        for k in range(p):
            V[j * p + k][col] = 1
            W[k * n + 0][col] = 1
            W[k * n + 1][col] = 0 # Explicitly 0, though initialized to 0
            
        # r=1 (col+1)
        # i=1
        U[1 * m + j][col + 1] = 1
        # i=0 is 0
        for k in range(p):
            V[j * p + k][col + 1] = 1
            W[k * n + 0][col + 1] = 0
            W[k * n + 1][col + 1] = 1
            
        col += 2
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
