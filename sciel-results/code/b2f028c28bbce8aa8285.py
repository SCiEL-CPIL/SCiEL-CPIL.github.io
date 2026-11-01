"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 2]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 7
    
    # U: 4x7
    U = [
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ]
    
    # V: 4x7
    V = [
        [1, 1, 0, 0, 0, 1, 0],
        [0, 0, 1, -1, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ]
    
    # W: 4x7
    # Rows correspond to k*n+i:
    # 0: k=0, i=0 -> C[0,0]
    # 1: k=0, i=1 -> C[1,0]
    # 2: k=1, i=0 -> C[0,1]
    # 3: k=1, i=1 -> C[1,1]
    W = [
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1, 0, -1, 1],
        [0, 0, 1, 0, 1, 1, -1],
        [1, -1, 1, 0, 0, 1, 0]
    ]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
