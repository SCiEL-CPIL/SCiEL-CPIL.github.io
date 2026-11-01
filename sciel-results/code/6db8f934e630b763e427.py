"""Rewritten program with Strassen's algorithm for rank-7 decomposition."""
DIMENSIONS = [2, 2, 2]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 7
    # U shape (4, 7), V shape (4, 7), W shape (4, 7)
    # Indices:
    # A: 00, 01, 10, 11 -> 0, 1, 2, 3
    # B: 00, 01, 10, 11 -> 0, 1, 2, 3
    # C: 00, 10, 01, 11 -> 0, 1, 2, 3 (based on W[k*n+i])
    
    U = [
        [1,  1,  1,  0,  1, -1,  0],  # A00
        [0,  0,  0,  0,  1,  0,  1],  # A01
        [0,  1,  0,  0,  0,  1,  0],  # A10
        [1,  1,  0,  1,  0,  0, -1]   # A11
    ]
    
    V = [
        [1,  1,  0, -1,  0,  1,  0],  # B00
        [0,  0,  1,  0,  0,  1,  0],  # B01
        [0,  0,  0,  1,  0,  0,  1],  # B10
        [1,  0, -1,  0,  1,  0,  1]   # B11
    ]
    
    W = [
        [1,  0,  0,  1, -1,  0,  1],  # C00 (idx 0)
        [0,  1,  0,  1,  0,  0,  0],  # C10 (idx 1)
        [0,  0,  1,  0,  1,  0,  0],  # C01 (idx 2)
        [1, -1,  1,  0,  0,  1,  0]   # C11 (idx 3)
    ]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
