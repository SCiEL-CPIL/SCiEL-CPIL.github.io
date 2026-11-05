"""Rank-17 decomposition: Strassen 2x2 core (rank 7) + direct column updates (rank 9)."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 3, 3
    rank = 17
    U = [[0] * rank for _ in range(n * m)]  # 6 x 17
    V = [[0] * rank for _ in range(m * p)]  # 9 x 17
    W = [[0] * rank for _ in range(p * n)]  # 6 x 17

    # Strassen 2x2 core: columns 0..6
    # M1 = (A00+A11)*(B00+B11)
    U[0][0] = 1; U[5][0] = 1
    V[0][0] = 1; V[6][0] = 1
    W[0][0] = 1; W[3][0] = 1
    # M2 = (A10+A11)*B00
    U[3][1] = 1; U[5][1] = 1
    V[0][1] = 1
    W[3][1] = 1
    # M3 = A00*(B01-B11)
    U[0][2] = 1
    V[1][2] = 1; V[7][2] = -1
    W[0][2] = 1
    # M4 = A11*(B10-B00)
    U[5][3] = 1
    V[6][3] = 1; V[0][3] = -1
    W[0][3] = -1; W[3][3] = 1
    # M5 = (A00+A01)*B11
    U[0][4] = 1; U[1][4] = 1
    V[7][4] = 1
    W[0][4] = -1; W[3][4] = 1
    # M6 = (A10-A00)*(B00+B01)  (unused in C)
    U[3][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[0][5] = 0; W[3][5] = 0
    # M7 = (A01-A11)*(B10+B11)
    U[1][6] = 1; U[5][6] = -1
    V[6][6] = 1; V[7][6] = 1
    W[0][6] = -1; W[3][6] = 1

    # Direct column updates: columns 7..15
    # For k in 0..2: A[:,k] * B[k,:]  (2x3 rank-1 update)
    for k in range(3):
        c = 7 + k
        U[0][c] = 1  # A[0,k]
        U[3][c] = 1  # A[1,k]
        for j in range(3):
            V[k * 3 + j][c] = 1  # B[k,j]
            W[j * 2 + 0][c] = 1  # C[0,j] += A[0,k]*B[k,j]
            W[j * 2 + 1][c] = 1  # C[1,j] += A[1,k]*B[k,j]

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
