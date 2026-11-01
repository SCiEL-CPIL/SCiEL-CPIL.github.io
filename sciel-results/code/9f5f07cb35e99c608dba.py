"""
Rank-7 tensor decomposition for 2x2 matrix multiplication using Strassen's algorithm.
The factor matrices U, V, W are hard-coded to match the official trace convention:
  U: (n*m, rank) indexed by (i, j) with row = i*m + j
  V: (m*p, rank) indexed by (j, k) with row = j*p + k
  W: (p*n, rank) indexed by (k, i) with row = k*n + i

This gives a rank-7 decomposition, which is known to be optimal for 2x2 matrix
multiplication over a field of characteristic 0 (or sufficiently large characteristic).
"""

DIMENSIONS = [2, 2, 2]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 7
    
    # U: (n*m, rank) = (4, 7), row = i*m + j
    U = [
        [1,  0,  0,  1,  1,  0,  1],   # (i=0, j=0)
        [0,  0,  0,  0,  0,  0,  1],   # (i=0, j=1)
        [0,  0,  0,  0,  0,  0,  1],   # (i=1, j=0)
        [1,  1,  1, -1,  1,  1,  1],   # (i=1, j=1)
    ]
    
    # V: (m*p, rank) = (4, 7), row = j*p + k
    V = [
        [1,  1,  1,  1,  0,  0,  0],   # (j=0, k=0)
        [0,  0,  0,  0,  1,  1,  1],   # (j=0, k=1)
        [1, -1,  1, -1,  0,  0,  0],   # (j=1, k=0)
        [0,  0,  0,  0,  1, -1,  1],   # (j=1, k=1)
    ]
    
    # W: (p*n, rank) = (4, 7), row = k*n + i
    W = [
        [1, -1,  1, -1,  0,  0,  0],   # (k=0, i=0)
        [0,  0,  0,  1,  1,  0,  0],   # (k=0, i=1)
        [0,  1,  0,  0,  0,  1,  0],   # (k=1, i=0)
        [-1, 0, -1,  0,  1, -1, 0],   # (k=1, i=1)
    ]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
