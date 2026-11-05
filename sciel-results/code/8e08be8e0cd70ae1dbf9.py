import time

DIMENSIONS = [3, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    # Target rank 82
    rank = 82
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_schoolbook(i, j, k, col):
        U[i * m + j][col] = 1
        V[j * p + k][col] = 1
        W[k * n + i][col] = 1

    def add_strassen(i0, i1, j0, j1, k0, k1, start_col):
        # Maps local indices (0,1) to global
        # M1
        U[i0 * m + j0][start_col + 0] = 1
        U[i1 * m + j1][start_col + 0] = 1
        V[j0 * p + k0][start_col + 0] = 1
        V[j1 * p + k1][start_col + 0] = 1
        W[k0 * n + i0][start_col + 0] = 1
        W[k1 * n + i1][start_col + 0] = 1
        
        # M2
        U[i1 * m + j0][start_col + 1] = 1
        U[i1 * m + j1][start_col + 1] = 1
        V[j0 * p + k0][start_col + 1] = 1
        W[k0 * n + i1][start_col + 1] = 1
        W[k1 * n + i1][start_col + 1] = -1
        
        # M3
        U[i0 * m + j0][start_col + 2] = 1
        V[j0 * p + k1][start_col + 2] = 1
        V[j1 * p + k1][start_col + 2] = -1
        W[k1 * n + i0][start_col + 2] = 1
        W[k1 * n + i1][start_col + 2] = 1
        
        # M4
        U[i1 * m + j1][start_col + 3] = 1
        V[j1 * p + k0][start_col + 3] = 1
        V[j0 * p + k0][start_col + 3] = -1
        W[k0 * n + i0][start_col + 3] = 1
        W[k0 * n + i1][start_col + 3] = 1
        
        # M5
        U[i0 * m + j0][start_col + 4] = 1
        U[i0 * m + j1][start_col + 4] = 1
        V[j1 * p + k1][start_col + 4] = 1
        W[k0 * n + i0][start_col + 4] = -1
        W[k1 * n + i0][start_col + 4] = 1
        
        # M6
        U[i1 * m + j0][start_col + 5] = 1
        U[i0 * m + j0][start_col + 5] = -1
        V[j0 * p + k0][start_col + 5] = 1
        V[j0 * p + k1][start_col + 5] = 1
        W[k1 * n + i1][start_col + 5] = 1
        
        # M7
        U[i0 * m + j1][start_col + 6] = 1
        U[i1 * m + j1][start_col + 6] = -1
        V[j1 * p + k0][start_col + 6] = 1
        V[j1 * p + k1][start_col + 6] = 1
        W[k0 * n + i0][start_col + 6] = 1

    # Block 1: i in {0,1}, j in {0,1}, k in {0,1}
    add_strassen(0, 1, 0, 1, 0, 1, 0)
    
    # Block 2: i in {0,1}, j in {2,3}, k in {2,3}
    add_strassen(0, 1, 2, 3, 2, 3, 7)
    
    # Schoolbook for the rest
    col = 14
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i in [0,1] and j in [0,1] and k in [0,1]):
                    continue
                if (i in [0,1] and j in [2,3] and k in [2,3]):
                    continue
                add_schoolbook(i, j, k, col)
                col += 1
                
    return {"U": U, "V": V, "W": W}
