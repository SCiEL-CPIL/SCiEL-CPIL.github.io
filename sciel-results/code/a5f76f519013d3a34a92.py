import time

DIMENSIONS = [2, 2, 3]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-minimal exact tensor decomposition for 2x2x3 matrix multiplication.
    Rank is 11.
    U: (4, 11), V: (6, 11), W: (6, 11)
    Indices:
    U[i*m + j, r]
    V[j*p + k, r]
    W[k*n + i, r]
    """
    n, m, p = DIMENSIONS
    rank = 11
    
    # Initialize with zeros
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add values
    def set_val(mat, i, j, r, val):
        mat[i][r] = val
        
    def add_val(mat, i, j, r, val):
        mat[i][r] += val

    # Strassen-like decomposition for the first 2 columns of B (2x2x2 part)
    # Plus 4 additional terms for the 3rd column of B.
    # Total rank 7 + 4 = 11.
    
    # The 2x2x2 Strassen decomposition (rank 7) for C[:, 0:2]
    # M1 = (A00+A11)(B00+B11)
    add_val(U, 0*2+0, 0, 0, 1)
    add_val(U, 1*2+1, 0, 0, 1)
    add_val(V, 0*3+0, 0, 0, 1)
    add_val(V, 1*3+1, 0, 0, 1)
    add_val(W, 0*2+0, 0, 0, 1) # C00
    add_val(W, 1*2+1, 0, 0, 1) # C11

    # M2 = (A10+A11)B00
    add_val(U, 1*2+0, 1, 1, 1)
    add_val(U, 1*2+1, 1, 1, 1)
    add_val(V, 0*3+0, 1, 1, 1)
    add_val(W, 0*2+1, 1, 1, 1) # C10
    add_val(W, 1*2+1, 1, 1, -1) # C11

    # M3 = A00(B01-B11)
    add_val(U, 0*2+0, 2, 1, 1)
    add_val(V, 0*3+1, 2, 1, 1)
    add_val(V, 1*3+1, 2, 1, -1)
    add_val(W, 1*2+0, 2, 1, 1) # C01
    add_val(W, 1*2+1, 2, 1, 1) # C11

    # M4 = A11(B10-B00)
    add_val(U, 1*2+1, 3, 1, 1)
    add_val(V, 1*3+0, 3, 1, 1)
    add_val(V, 0*3+0, 3, 1, -1)
    add_val(W, 0*2+0, 3, 1, 1) # C00
    add_val(W, 0*2+1, 3, 1, 1) # C10

    # M5 = (A00+A01)B11
    add_val(U, 0*2+0, 4, 1, 1)
    add_val(U, 0*2+1, 4, 1, 1)
    add_val(V, 1*3+1, 4, 1, 1)
    add_val(W, 0*2+0, 4, 1, -1) # C00
    add_val(W, 1*2+0, 4, 1, 1) # C01

    # M6 = (A10-A00)(B00+B01)
    add_val(U, 1*2+0, 5, 1, 1)
    add_val(U, 0*2+0, 5, 1, -1)
    add_val(V, 0*3+0, 5, 1, 1)
    add_val(V, 0*3+1, 5, 1, 1)
    add_val(W, 1*2+1, 5, 1, 1) # C11

    # M7 = (A01-A11)(B10+B11)
    add_val(U, 0*2+1, 6, 1, 1)
    add_val(U, 1*2+1, 6, 1, -1)
    add_val(V, 1*3+0, 6, 1, 1)
    add_val(V, 1*3+1, 6, 1, 1)
    add_val(W, 0*2+0, 6, 1, 1) # C00

    # Additional terms for the 3rd column of B (k=2)
    # C02 = A00*B02 + A01*B12
    # C12 = A10*B02 + A11*B12
    
    # r=7: A00 * B02
    add_val(U, 0*2+0, 7, 1, 1)
    add_val(V, 0*3+2, 7, 1, 1)
    add_val(W, 2*2+0, 7, 1, 1) # C02

    # r=8: A01 * B12
    add_val(U, 0*2+1, 8, 1, 1)
    add_val(V, 1*3+2, 8, 1, 1)
    add_val(W, 2*2+0, 8, 1, 1) # C02

    # r=9: A10 * B02
    add_val(U, 1*2+0, 9, 1, 1)
    add_val(V, 0*3+2, 9, 1, 1)
    add_val(W, 2*2+1, 9, 1, 1) # C12

    # r=10: A11 * B12
    add_val(U, 1*2+1, 10, 1, 1)
    add_val(V, 1*3+2, 10, 1, 1)
    add_val(W, 2*2+1, 10, 1, 1) # C12

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
