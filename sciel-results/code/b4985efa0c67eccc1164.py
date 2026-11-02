"""Strassen rank-7 exact decomposition for 2x2x2 matrix multiplication."""
DIMENSIONS = [2, 2, 2]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 7

    # U has shape (n*m, rank) = (4, 7)
    # Index: row = i*m + j  =>  i*m + j
    # V has shape (m*p, rank) = (4, 7)
    # Index: row = j*p + k
    # W has shape (p*n, rank) = (4, 7)
    # Index: row = k*n + i   (official trace convention W[k*n+i, r])

    # Strassen products:
    # P1 = (A00+A11)(B00+B11)
    # P2 = (A10+A11)B00
    # P3 = A00(B01-B11)
    # P4 = A11(B10-B00)
    # P5 = (A00+A10)B11
    # P6 = (A10-A00)(B00+B01)
    # P7 = (A01-A11)(B10+B11)
    #
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6

    U = [
        [1,  0,  1,  0,  1, -1,  0],  # A00 (row 0)
        [0,  0,  0,  1,  0,  0, -1],  # A01 (row 1)
        [0,  1,  0,  0,  0,  1,  0],  # A10 (row 2)
        [1,  1,  0,  0,  0,  0,  0],  # A11 (row 3)
    ]

    V = [
        [1,  1,  0, -1,  0,  1,  0],  # B00 (row 0)
        [1,  0,  1,  0,  0,  1,  1],  # B01 (row 1)
        [0,  0, -1,  1,  0,  0,  0],  # B10 (row 2)
        [1,  0,  0,  0,  1,  0,  0],  # B11 (row 3)
    ]

    # W rows indexed by k*n + i
    # k=0, i=0 => C00 => P1+P4-P5+P7
    # k=0, i=1 => C01 => P3+P5
    # k=1, i=0 => C10 => P2+P4
    # k=1, i=1 => C11 => P1-P2+P3+P6
    W = [
        [1,  0,  0,  1, -1,  0,  1],  # C00
        [0,  0,  1,  0,  1,  0,  0],  # C01
        [0,  1,  0,  1,  0,  0,  0],  # C10
        [1, -1,  1,  0,  0,  1,  0],  # C11
    ]

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
