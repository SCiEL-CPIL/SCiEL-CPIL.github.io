"""Strassen rank-7 exact tensor decomposition for 2x2x2 matrix multiplication."""
DIMENSIONS = [2, 2, 2]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns the rank-7 Strassen decomposition for 2x2 matrix multiplication.
    This is the proven optimal rank.
    """
    # U: shape (4, 7) -> rows correspond to input A indices (0,1,2,3) for A[0,0], A[0,1], A[1,0], A[1,1]
    # V: shape (4, 7) -> rows correspond to input B indices (0,1,2,3) for B[0,0], B[0,1], B[1,0], B[1,1]
    # W: shape (4, 7) -> rows correspond to output C indices (0,1,2,3) for C[0,0], C[0,1], C[1,0], C[1,1]
    # Note: The task description says W[k*n+i, r].
    # Standard matrix multiplication C[i,k] = sum_j A[i,j] B[j,k].
    # Indices for A: i*m + j. m=2.
    # i=0, j=0 -> 0; i=0, j=1 -> 1; i=1, j=0 -> 2; i=1, j=1 -> 3.
    # Indices for B: j*p + k. p=2.
    # j=0, k=0 -> 0; j=0, k=1 -> 1; j=1, k=0 -> 2; j=1, k=1 -> 3.
    # Indices for C: k*n + i. n=2.
    # k=0, i=0 -> 0; k=0, i=1 -> 1; k=1, i=0 -> 2; k=1, i=1 -> 3.
    # So C[0,0] is row 0, C[1,0] is row 1, C[0,1] is row 2, C[1,1] is row 3.
    
    U = [
        [1, 0, 1, 0, 1, -1, 0],  # A[0,0] (a)
        [0, 0, 0, 0, 1, 0, 1],   # A[0,1] (b)
        [0, 1, 0, 0, 0, 1, 0],   # A[1,0] (c)
        [1, 1, 0, 1, 0, 0, -1]   # A[1,1] (d)
    ]
    
    V = [
        [1, 1, 0, -1, 0, 1, 0],  # B[0,0] (e)
        [0, 0, 1, 0, 0, 1, 0],   # B[0,1] (f)
        [0, 0, 0, 1, 0, 0, 1],   # B[1,0] (g)
        [1, 0, -1, 0, 1, 0, 1]   # B[1,1] (h)
    ]
    
    W = [
        [1, 0, 0, 1, -1, 0, 1],  # C[0,0]
        [0, 1, 0, 1, 0, 0, 0],   # C[1,0]
        [0, 0, 1, 0, 1, 0, 0],   # C[0,1]
        [1, -1, 1, 0, 0, 1, 0]   # C[1,1]
    ]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
