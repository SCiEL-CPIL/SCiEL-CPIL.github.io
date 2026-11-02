"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 2]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 7
    
    # Strassen rank-7 decomposition
    # U shape (n*m, rank) -> (4, 7)
    # V shape (m*p, rank) -> (4, 7)
    # W shape (p*n, rank) -> (4, 7)
    
    U = [
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ]
    
    V = [
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ]
    
    W = [
        [1, 0, 0, -1, 1, 0, 1],
        [0, 1, 0, 1, 0, 1, 1],
        [0, 0, 1, 1, 0, -1, -1],
        [1, 0, 1, 0, 1, 0, 0]
    ]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
