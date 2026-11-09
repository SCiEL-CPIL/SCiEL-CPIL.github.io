"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 28 decomposition
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: Top row (i=0)
    # R(1, 4, 6) = 4
    # c = 0, 1, 2, 3
    for c in range(4):
        # U[0, c] = 1 (i=0, j=c)
        U[0 * m + c][col] = 1
        # V[j, k, c] = 1 if j=c
        for k in range(p):
            V[c * p + k][col] = 1
        # W[k, 0, c] = 1
        for k in range(p):
            W[k * n + 0][col] = 1
        col += 1
        
    # Part 2: Bottom rows (i=1, 2)
    # Split k into 3 groups of 2: [0,1], [2,3], [4,5]
    # For each group, R(2, 4, 2) = 8
    # R(2, 4, 2) splits into two R(1, 4, 2) for i=1 and i=2
    # Each R(1, 4, 2) has rank 4.
    
    for group in range(3):
        k_start = group * 2
        k_end = k_start + 2
        # For i=1 and i=2
        for i_offset in range(2):
            i = 1 + i_offset
            # R(1, 4, 2) rank 4
            for c in range(4):
                # U[i, c] = 1 (i=1 or 2, j=c)
                U[i * m + c][col] = 1
                # V[j, k, c] = 1 if j=c and k in [k_start, k_end]
                for k in range(k_start, k_end):
                    V[c * p + k][col] = 1
                # W[k, i, c] = 1 for k in [k_start, k_end]
                for k in range(k_start, k_end):
                    W[k * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
