"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 2]

# EVOLVE-BLOCK-START
def construct():
    # Strassen decomposition of rank 7 for 2x2 matrix multiplication.
    n, m, p = DIMENSIONS
    rank = 7
    
    # U shape: (n*m, rank) = (4, 7)
    # V shape: (m*p, rank) = (4, 7)
    # W shape: (p*n, rank) = (4, 7)
    
    # Indices for U: i*m + j
    # Indices for V: j*p + k
    # Indices for W: k*n + i
    
    U = [
        [1, 0, 1, 0, 1, -1, 0],  # a
        [0, 0, 0, 0, 1, 0, 1],   # b
        [0, 1, 0, 0, 0, 1, 0],   # c
        [1, 1, 0, 1, 0, 0, -1]   # d
    ]
    
    V = [
        [1, 1, 0, 0, 0, 1, 0],   # e
        [1, 0, 1, -1, 0, 1, 0],  # f
        [0, 0, 0, 1, 0, 0, 1],   # g
        [0, 0, -1, 0, 1, 0, 1]   # h
    ]
    
    W = [
        [1, 0, 0, 1, -1, 0, 1],  # p
        [0, 0, 1, 0, 1, 0, 0],   # q
        [0, 1, 0, 1, 0, 0, 0],   # r
        [1, -1, 1, 0, 0, 1, 0]   # s
    ]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
