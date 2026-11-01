"""Strassen-based rank-21 decomposition for 2x2 * 2x6 matrix multiplication."""
DIMENSIONS = [2, 2, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # rank = 7 per block of 2 columns in B, 3 blocks
    rank = 21

    # U: (n*m, rank) = (4, 21)
    # Rows indexed by i*m+j -> A[i,j]
    # V: (m*p, rank) = (12, 21)
    # Rows indexed by j*p+k -> B[j,k]
    # W: (p*n, rank) = (12, 21)
    # Rows indexed by k*n+i -> C[i,k]

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # For each block b (0,1,2), we have 7 rank-1 terms
    # Block b handles columns 2b and 2b+1 of B and C
    #
    # The 7 Strassen terms for a 2x2 * 2x2 block:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11) B01
    # M3 = A00 (B10 - B11)
    # M4 = A11 (B00 - B01)
    # M5 = (A00+A01) B11
    # M6 = (A10-A00)(B00+B11)
    # M7 = (A01-A11)(B10+B11)
    #
    # Outputs:
    # C00 = M1 - M4 + M5 - M7
    # C01 = M3 + M4 + M7
    # C10 = M2 + M3 - M5 + M6
    # C11 = M1 - M2 + M5 - M6

    for b in range(3):
        base = b * 7  # starting column index in U, V, W
        col0 = 2 * b      # even column of B
        col1 = 2 * b + 1  # odd column of B

        # M1: (A00+A11)(B00+B11)
        c = base + 0
        U[0 * m + 0][c] += 1   # A[0,0]
        U[1 * m + 1][c] += 1   # A[1,1]
        V[0 * p + col0][c] += 1   # B[0, col0]
        V[1 * p + col1][c] += 1   # B[1, col1]
        # C[0, col0] = M1 - M4 + M5 - M7  -> coefficient +1
        W[col0 * n + 0][c] += 1
        # C[1, col1] = M1 - M2 + M5 - M6  -> coefficient +1
        W[col1 * n + 1][c] += 1

        # M2: (A10+A11) B01
        c = base + 1
        U[1 * m + 0][c] += 1   # A[1,0]
        U[1 * m + 1][c] += 1   # A[1,1]
        V[0 * p + col1][c] += 1   # B[0, col1]
        # C[1, col0] = M2 + M3 - M5 + M6  -> coefficient +1
        W[col0 * n + 1][c] += 1
        # C[1, col1] = M1 - M2 + M5 - M6  -> coefficient -1
        W[col1 * n + 1][c] += -1

        # M3: A00 (B10 - B11)
        c = base + 2
        U[0 * m + 0][c] += 1   # A[0,0]
        V[1 * p + col0][c] += 1   # B[1, col0]
        V[1 * p + col1][c] += -1  # B[1, col1]
        # C[0, col1] = M3 + M4 + M7  -> coefficient +1
        W[col1 * n + 0][c] += 1
        # C[1, col0] = M2 + M3 - M5 + M6  -> coefficient +1
        W[col0 * n + 1][c] += 1

        # M4: A11 (B00 - B01)
        c = base + 3
        U[1 * m + 1][c] += 1   # A[1,1]
        V[0 * p + col0][c] += 1   # B[0, col0]
        V[0 * p + col1][c] += -1  # B[0, col1]
        # C[0, col0] = M1 - M4 + M5 - M7  -> coefficient -1
        W[col0 * n + 0][c] += -1
        # C[0, col1] = M3 + M4 + M7  -> coefficient +1
        W[col1 * n + 0][c] += 1

        # M5: (A00+A01) B11
        c = base + 4
        U[0 * m + 0][c] += 1   # A[0,0]
        U[0 * m + 1][c] += 1   # A[0,1]
        V[1 * p + col1][c] += 1   # B[1, col1]
        # C[0, col0] = M1 - M4 + M5 - M7  -> coefficient +1
        W[col0 * n + 0][c] += 1
        # C[0, col1] = M3 + M4 + M7  -> coefficient 0 (not here)
        # C[1, col0] = M2 + M3 - M5 + M6  -> coefficient -1
        W[col0 * n + 1][c] += -1
        # C[1, col1] = M1 - M2 + M5 - M6  -> coefficient +1
        W[col1 * n + 1][c] += 1

        # M6: (A10-A00)(B00+B11)
        c = base + 5
        U[1 * m + 0][c] += 1   # A[1,0]
        U[0 * m + 0][c] += -1  # A[0,0]
        V[0 * p + col0][c] += 1   # B[0, col0]
        V[1 * p + col1][c] += 1   # B[1, col1]
        # C[1, col0] = M2 + M3 - M5 + M6  -> coefficient +1
        W[col0 * n + 1][c] += 1
        # C[1, col1] = M1 - M2 + M5 - M6  -> coefficient -1
        W[col1 * n + 1][c] += -1

        # M7: (A01-A11)(B10+B11)
        c = base + 6
        U[0 * m + 1][c] += 1   # A[0,1]
        U[1 * m + 1][c] += -1  # A[1,1]
        V[1 * p + col0][c] += 1   # B[1, col0]
        V[1 * p + col1][c] += 1   # B[1, col1]
        # C[0, col0] = M1 - M4 + M5 - M7  -> coefficient -1
        W[col0 * n + 0][c] += -1
        # C[0, col1] = M3 + M4 + M7  -> coefficient +1
        W[col1 * n + 0][c] += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
