import numpy as np

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=4, m=5, p=5
    # U: (n*m, rank) = (20, rank)
    # V: (m*p, rank) = (25, rank)
    # W: (p*n, rank) = (20, rank)
    
    rank = 19
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Part 1: A_1 B_11 (4x4 x 4x4) -> Rank 16
    # A_1: rows 0..3, cols 0..3 of A (4x5)
    # B_11: rows 0..3, cols 0..3 of B (5x5)
    # C_11: rows 0..3, cols 0..3 of C (4x5)
    r = 0
    for i in range(4):
        for j in range(4):
            for k in range(4):
                U[i * m + j, r] = 1
                V[j * p + k, r] = 1
                W[k * n + i, r] = 1
                r += 1
                
    # Part 2: A_2 B_21 (4x1 x 1x4) -> Rank 1
    # A_2: rows 0..3, col 4 of A
    # B_21: row 4, cols 0..3 of B
    # C_21: rows 0..3, cols 0..3 of C
    r = 16
    for i in range(4):
        for k in range(4):
            U[i * m + 4, r] = 1
            V[4 * p + k, r] = 1
            W[k * n + i, r] = 1
            
    # Part 3: A_1 B_12 (4x4 x 4x1) -> Rank 1
    # A_1: rows 0..3, cols 0..3 of A
    # B_12: rows 0..3, col 4 of B
    # C_12: rows 0..3, col 4 of C
    r = 17
    for i in range(4):
        for j in range(4):
            U[i * m + j, r] = 1
            V[j * p + 4, r] = 1
            W[4 * n + i, r] = 1
            
    # Part 4: A_2 B_22 (4x1 x 1x1) -> Rank 1
    # A_2: rows 0..3, col 4 of A
    # B_22: row 4, col 4 of B
    # C_22: rows 0..3, col 4 of C
    r = 18
    for i in range(4):
        U[i * m + 4, r] = 1
        V[4 * p + 4, r] = 1
        W[4 * n + i, r] = 1
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
