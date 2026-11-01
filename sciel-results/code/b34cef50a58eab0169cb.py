"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 2]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 7

    # Strassen's algorithm rank-7 decomposition for 2x2 matrix multiplication.
    #
    # Conventions:
    #   U has shape (n*m, rank) = (4, 7).  Row index r = i*m + j  (i row of A, j col of A).
    #   V has shape (m*p, rank) = (4, 7).  Row index s = j*p + k  (j row of B, k col of B).
    #   W has shape (p*n, rank) = (4, 7).  Row index t = k*n + i  (k col of C, i row of C).
    #
    # The seven scalar products (columns of U,V,W) are:
    #   c0 = (A00 + A11)(B00 + B11)
    #   c1 = (A10 + A11) * B00
    #   c2 =  A00         * (B10 - B00)
    #   c3 =  A11         * (B11 - B01)
    #   c4 = (A00 + A01)  * B11
    #   c5 = (A10 - A00)  * (B00 + B01)
    #   c6 = (A01 - A11)  * (B10 + B11)
    #
    # Output reconstruction (W convention W[k*n + i]):
    #   C00 = c0 - c3 + c5 + c6   -> row t=0 (k=0,i=0)
    #   C01 = c4 + c6             -> row t=2 (k=1,i=0)
    #   C10 = c1 + c5             -> row t=1 (k=0,i=1)
    #   C11 = c0 - c2 + c1 + c4   -> row t=3 (k=1,i=1)

    U = [
        [1, 0, 1, 0, 1, -1, 0],   # A00 (i=0,j=0)
        [0, 0, 0, 0, 1,  0, 1],   # A01 (i=0,j=1)
        [0, 1, 0, 0, 0,  1, 0],   # A10 (i=1,j=0)
        [1, 1, 0, 1, 0,  0,-1],   # A11 (i=1,j=1)
    ]

    V = [
        [1, 1, -1, 0, 0, 1, 0],   # B00 (j=0,k=0)
        [1, 0,  0, 0, 0, 1, 0],   # B01 (j=0,k=1)
        [1, 0,  1, 0, 0, 0, 1],   # B10 (j=1,k=0)
        [1, 0,  0, 1, 1, 0, 1],   # B11 (j=1,k=1)
    ]

    W = [
        [1, 0, 0, -1, 0, 1, 1],   # C00 (k=0,i=0)
        [0, 1, 0,  0, 0, 1, 0],   # C10 (k=0,i=1)
        [0, 0, 0,  0, 1, 0, 1],   # C01 (k=1,i=0)
        [1, 1, -1, 0, 1, 0, 0],   # C11 (k=1,i=1)
    ]

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
