"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 2]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 7
    
    # U shape (n*m, rank) = (4, 7)
    # V shape (m*p, rank) = (4, 7)
    # W shape (p*n, rank) = (4, 7)
    
    # Strassen's algorithm decomposition
    # Indices for A (n*m): 0:a, 1:b, 2:c, 3:d
    # Indices for B (m*p): 0:e, 1:f, 2:g, 3:h
    # Indices for C (p*n): 0:C00, 1:C01, 2:C10, 3:C11
    
    U = [
        [1, 0, 1, 0, 1, -1, 0],  # r0..r6 for a
        [0, 0, 0, 0, 1, 0, 1],   # for b
        [0, 1, 0, 0, 0, 1, 0],   # for c
        [1, 1, 0, 1, 0, 0, -1]   # for d
    ]
    
    V = [
        [1, 1, 0, -1, 0, 1, 0],  # r0..r6 for e
        [0, 0, 1, 0, 0, 1, 0],   # for f
        [0, 0, 0, 1, 0, 0, 1],   # for g
        [1, 0, -1, 0, 1, 0, 1]   # for h
    ]
    
    W = [
        [1, 0, 0, 0, -1, 0, 1],  # r0..r6 for C00
        [1, 0, 1, 0, 1, 0, 0],   # for C01
        [1, 1, 0, 1, 0, 0, 0],   # for C10
        [1, 1, 0, 1, 0, 1, 0]    # for C11
    ]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
