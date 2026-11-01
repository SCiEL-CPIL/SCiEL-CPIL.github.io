"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 19 decomposition
    # Block 0: M(2,2,2) for j in {0,1}, k in {0,1} -> Rank 7
    # Block 1: M(2,2,2) for j in {0,1}, k in {2,3} -> Rank 7
    # Block 2: M(2,2,1) for j in {0,1}, k in {4}   -> Rank 4
    # Block 3: M(2,1,5) for j in {2},   k in {0..4} -> Rank 1
    rank = 19
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    def add_strassen(col_offset, i_offset, j_offset, k_offset):
        # i_offset, j_offset, k_offset are global start indices
        # Local i, j, k in 0..1
        # U index: (i_offset + i) * m + (j_offset + j)
        # V index: (j_offset + j) * p + (k_offset + k)
        # W index: (k_offset + k) * n + (i_offset + i)
        
        # M1
        U[(i_offset + 0) * m + (j_offset + 0)][col_offset + 0] = 1
        U[(i_offset + 1) * m + (j_offset + 1)][col_offset + 0] = 1
        V[(j_offset + 0) * p + (k_offset + 0)][col_offset + 0] = 1
        V[(j_offset + 1) * p + (k_offset + 1)][col_offset + 0] = 1
        W[(k_offset + 0) * n + (i_offset + 0)][col_offset + 0] = 1
        W[(k_offset + 1) * n + (i_offset + 1)][col_offset + 0] = 1

        # M2
        U[(i_offset + 1) * m + (j_offset + 0)][col_offset + 1] = 1
        U[(i_offset + 1) * m + (j_offset + 1)][col_offset + 1] = 1
        V[(j_offset + 0) * p + (k_offset + 0)][col_offset + 1] = 1
        W[(k_offset + 0) * n + (i_offset + 1)][col_offset + 1] = 1

        # M3
        U[(i_offset + 0) * m + (j_offset + 0)][col_offset + 2] = 1
        V[(j_offset + 0) * p + (k_offset + 1)][col_offset + 2] = 1
        V[(j_offset + 1) * p + (k_offset + 1)][col_offset + 2] = -1
        W[(k_offset + 1) * n + (i_offset + 0)][col_offset + 2] = 1

        # M4
        U[(i_offset + 1) * m + (j_offset + 1)][col_offset + 3] = 1
        V[(j_offset + 1) * p + (k_offset + 0)][col_offset + 3] = 1
        V[(j_offset + 0) * p + (k_offset + 0)][col_offset + 3] = -1
        W[(k_offset + 0) * n + (i_offset + 0)][col_offset + 3] = 1
        W[(k_offset + 0) * n + (i_offset + 1)][col_offset + 3] = 1

        # M5
        U[(i_offset + 0) * m + (j_offset + 0)][col_offset + 4] = 1
        U[(i_offset + 0) * m + (j_offset + 1)][col_offset + 4] = 1
        V[(j_offset + 1) * p + (k_offset + 1)][col_offset + 4] = 1
        W[(k_offset + 0) * n + (i_offset + 0)][col_offset + 4] = -1
        W[(k_offset + 1) * n + (i_offset + 0)][col_offset + 4] = 1

        # M6
        U[(i_offset + 1) * m + (j_offset + 0)][col_offset + 5] = 1
        U[(i_offset + 0) * m + (j_offset + 0)][col_offset + 5] = -1
        V[(j_offset + 0) * p + (k_offset + 0)][col_offset + 5] = 1
        V[(j_offset + 0) * p + (k_offset + 1)][col_offset + 5] = 1
        W[(k_offset + 1) * n + (i_offset + 1)][col_offset + 5] = 1

        # M7
        U[(i_offset + 0) * m + (j_offset + 1)][col_offset + 6] = 1
        U[(i_offset + 1) * m + (j_offset + 1)][col_offset + 6] = -1
        V[(j_offset + 1) * p + (k_offset + 0)][col_offset + 6] = 1
        V[(j_offset + 1) * p + (k_offset + 1)][col_offset + 6] = 1
        W[(k_offset + 0) * n + (i_offset + 0)][col_offset + 6] = 1
        W[(k_offset + 1) * n + (i_offset + 1)][col_offset + 6] = -1

    # Block 0: i=0..1, j=0..1, k=0..1
    add_strassen(0, 0, 0, 0)

    # Block 1: i=0..1, j=0..1, k=2..3
    add_strassen(7, 0, 0, 2)

    # Block 2: i=0..1, j=0..1, k=4 (Standard M(2,2,1))
    col = 14
    # Term 0: A(0,0)B(0,4) -> C(0,4)
    U[0 * m + 0][col] = 1
    V[0 * p + 4][col] = 1
    W[4 * n + 0][col] = 1
    col += 1
    # Term 1: A(0,1)B(1,4) -> C(0,4)
    U[0 * m + 1][col] = 1
    V[1 * p + 4][col] = 1
    W[4 * n + 0][col] = 1
    col += 1
    # Term 2: A(1,0)B(0,4) -> C(1,4)
    U[1 * m + 0][col] = 1
    V[0 * p + 4][col] = 1
    W[4 * n + 1][col] = 1
    col += 1
    # Term 3: A(1,1)B(1,4) -> C(1,4)
    U[1 * m + 1][col] = 1
    V[1 * p + 4][col] = 1
    W[4 * n + 1][col] = 1

    # Block 3: i=0..1, j=2, k=0..4 (Rank 1 M(2,1,5))
    col = 18
    for i in range(2):
        U[i * m + 2][col] = 1
    for k in range(5):
        V[2 * p + k][col] = 1
    for k in range(5):
        for i in range(2):
            W[k * n + i][col] = 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
