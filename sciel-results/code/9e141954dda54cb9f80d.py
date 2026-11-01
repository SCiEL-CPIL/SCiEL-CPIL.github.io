"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0

    def add_strassen_222(i_off, j_off, k_off):
        nonlocal col
        U[i_off * m + j_off][col] = 1
        U[(i_off+1) * m + (j_off+1)][col] = 1
        V[j_off * p + k_off][col] = 1
        V[(j_off+1) * p + (k_off+1)][col] = 1
        W[k_off * n + i_off][col] = 1
        W[(k_off+1) * n + (i_off+1)][col] = 1
        col += 1

        U[(i_off+1) * m + j_off][col] = 1
        U[(i_off+1) * m + (j_off+1)][col] = 1
        V[j_off * p + k_off][col] = 1
        W[k_off * n + (i_off+1)][col] = 1
        col += 1

        U[i_off * m + j_off][col] = 1
        V[(j_off+1) * p + k_off][col] = 1
        V[(j_off+1) * p + (k_off+1)][col] = -1
        W[k_off * n + i_off][col] = 1
        W[(k_off+1) * n + i_off][col] = -1
        col += 1

        U[(i_off+1) * m + (j_off+1)][col] = 1
        V[j_off * p + (k_off+1)][col] = 1
        V[j_off * p + k_off][col] = -1
        W[(k_off+1) * n + (i_off+1)][col] = 1
        W[k_off * n + (i_off+1)][col] = -1
        col += 1

        U[i_off * m + j_off][col] = 1
        U[i_off * m + (j_off+1)][col] = 1
        V[(j_off+1) * p + (k_off+1)][col] = 1
        W[(k_off+1) * n + i_off][col] = 1
        col += 1

        U[(i_off+1) * m + j_off][col] = 1
        U[i_off * m + j_off][col] = -1
        V[j_off * p + k_off][col] = 1
        V[(j_off+1) * p + k_off][col] = 1
        W[k_off * n + (i_off+1)][col] = 1
        W[k_off * n + i_off][col] = -1
        col += 1

        U[i_off * m + (j_off+1)][col] = 1
        U[(i_off+1) * m + (j_off+1)][col] = -1
        V[j_off * p + (k_off+1)][col] = 1
        V[(j_off+1) * p + (k_off+1)][col] = 1
        W[(k_off+1) * n + i_off][col] = 1
        W[(k_off+1) * n + (i_off+1)][col] = -1
        col += 1

    def add_direct_221(i_off, j_off, k_off):
        nonlocal col
        for i in range(2):
            for j in range(2):
                U[(i_off+i) * m + (j_off+j)][col] = 1
                V[(j_off+j) * p + k_off][col] = 1
                W[k_off * n + (i_off+i)][col] = 1
                col += 1

    def add_direct_212(i_off, j_off, k_off):
        nonlocal col
        for i in range(2):
            for k in range(2):
                U[(i_off+i) * m + j_off][col] = 1
                V[j_off * p + (k_off+k)][col] = 1
                W[(k_off+k) * n + (i_off+i)][col] = 1
                col += 1

    # Block A: j in 0..1, k in 0..2
    add_direct_221(0, 0, 0)
    add_strassen_222(0, 0, 1)
    
    # Block B: j in 0..1, k in 3..4
    add_strassen_222(0, 0, 3)
    
    # Block C: j=2, k in 0..2
    for i in range(2):
        for k in range(3):
            U[i * m + 2][col] = 1
            V[2 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1

    # Block D: j=2, k in 3..4
    add_direct_212(0, 2, 3)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
