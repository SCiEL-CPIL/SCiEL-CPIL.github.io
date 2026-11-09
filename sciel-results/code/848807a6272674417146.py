import time

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 22
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen(i0, i1, j0, j1, k0, k1):
        nonlocal col
        # M1
        U[i0 * m + j0][col] = 1
        U[(i1-1) * m + (j1-1)][col] = 1
        V[j0 * p + k0][col] = 1
        V[(j1-1) * p + (k1-1)][col] = 1
        W[k0 * n + i0][col] = 1
        W[(k1-1) * n + (i1-1)][col] = 1
        col += 1
        # M2
        U[(i1-1) * m + j0][col] = 1
        U[(i1-1) * m + (j1-1)][col] = 1
        V[j0 * p + k0][col] = 1
        W[k0 * n + (i1-1)][col] = 1
        W[(k1-1) * n + (i1-1)][col] = -1
        col += 1
        # M3
        U[i0 * m + j0][col] = 1
        V[j0 * p + (k1-1)][col] = 1
        V[(j1-1) * p + (k1-1)][col] = -1
        W[(k1-1) * n + i0][col] = 1
        W[(k1-1) * n + (i1-1)][col] = 1
        col += 1
        # M4
        U[(i1-1) * m + (j1-1)][col] = 1
        V[(j1-1) * p + k0][col] = 1
        V[j0 * p + k0][col] = -1
        W[k0 * n + i0][col] = 1
        W[k0 * n + (i1-1)][col] = 1
        col += 1
        # M5
        U[i0 * m + j0][col] = 1
        U[i0 * m + (j1-1)][col] = 1
        V[(j1-1) * p + (k1-1)][col] = 1
        W[k0 * n + i0][col] = -1
        W[(k1-1) * n + i0][col] = 1
        col += 1
        # M6
        U[(i1-1) * m + j0][col] = 1
        U[i0 * m + j0][col] = -1
        V[j0 * p + k0][col] = 1
        V[j0 * p + (k1-1)][col] = 1
        W[(k1-1) * n + (i1-1)][col] = 1
        col += 1
        # M7
        U[i0 * m + (j1-1)][col] = 1
        U[(i1-1) * m + (j1-1)][col] = -1
        V[(j1-1) * p + k0][col] = 1
        V[(j1-1) * p + (k1-1)][col] = 1
        W[k0 * n + i0][col] = 1
        W[k0 * n + (i1-1)][col] = -1
        col += 1

    add_strassen(0, 2, 0, 2, 0, 2)
    add_strassen(0, 2, 0, 2, 2, 4)
    
    for i in range(2):
        for k in range(2):
            U[i * m + 2][col] = 1
            V[2 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
            
    for i in range(2):
        for k in range(2, 4):
            U[i * m + 2][col] = 1
            V[2 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
