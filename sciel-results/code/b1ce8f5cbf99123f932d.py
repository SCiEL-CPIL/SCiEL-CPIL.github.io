"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We decompose M_{2,3,4} into M_{2,2,4} (rank 8) and M_{2,1,4} (rank 8).
    # Total rank = 16.
    
    rank1 = 8 # for 2x2x4 part
    rank2 = 8 # for 2x1x4 part
    total_rank = rank1 + rank2
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Part 1: j in {0, 1} (indices 0, 1 in the middle dimension)
    # A is 2x2 (rows 0,1 of A), B is 2x4 (rows 0,1 of B)
    # This is standard 2x2x4 multiplication.
    # U index: i*m + j. i in 0..1, j in 0..1.
    # V index: j*p + k. j in 0..1, k in 0..3.
    # W index: k*n + i. k in 0..3, i in 0..1.
    
    col = 0
    for i in range(2):
        for k in range(4):
            for j in range(2):
                # U[i, j] = 1
                # V[j, k] = 1
                # W[k, i] = 1
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    # Part 2: j in {2} (index 2 in the middle dimension)
    # A is 2x1 (row 2 of A), B is 1x4 (row 2 of B)
    # This is standard 2x1x4 multiplication.
    # U index: i*m + 2. i in 0..1.
    # V index: 2*p + k. k in 0..3.
    # W index: k*n + i. k in 0..3, i in 0..1.
    
    for i in range(2):
        for k in range(4):
            # U[i, 2] = 1
            # V[2, k] = 1
            # W[k, i] = 1
            U[i * m + 2][col] = 1
            V[2 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
