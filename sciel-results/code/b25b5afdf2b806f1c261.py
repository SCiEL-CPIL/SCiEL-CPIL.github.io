"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 2]

# EVOLVE-BLOCK-START
def construct():
    # Strassen decomposition of rank 7 for 2x2 matrix multiplication.
    n, m, p = DIMENSIONS
    rank = 7
    
    # U[i*m + j, r]
    U = [
        [1, 0, 1, 0, 1, 1, 0], # i=0, j=0 (A00)
        [0, 0, 0, 0, 1, 0, 1], # i=0, j=1 (A01)
        [0, 1, 0, 0, 0, 1, 0], # i=1, j=0 (A10)
        [1, 1, 0, 1, 0, 0, 1]  # i=1, j=1 (A11)
    ]
    
    # V[j*p + k, r]
    V = [
        [1, 1, 0, -1, 0, 0, 0], # j=0, k=0 (B00)
        [0, 0, 1, 0, 0, 1, 1],  # j=0, k=1 (B01)
        [0, 0, 0, 1, 0, 0, 1],  # j=1, k=0 (B10)
        [1, 0, -1, 0, 1, 0, 0]  # j=1, k=1 (B11)
    ]
    
    # W[k*n + i, r]
    # k=0, i=0 -> C00 (row 0)
    # k=0, i=1 -> C10 (row 1)
    # k=1, i=0 -> C01 (row 2)
    # k=1, i=1 -> C11 (row 3)
    W = [
        [1, 0, 0, -1, 1, 0, 1], # C00
        [1, 1, 0, 0, 0, 0, 0],  # C10
        [0, 0, 1, 1, 0, 0, 0],  # C01
        [0, -1, 0, 1, 0, 1, 1]  # C11
    ]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
