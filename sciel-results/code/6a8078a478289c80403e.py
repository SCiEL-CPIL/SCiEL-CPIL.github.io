import time

DIMENSIONS = [3, 4, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    n, m, p = DIMENSIONS
    rank = 58
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_schoolbook(i_start, i_end, j_start, j_end, k_start, k_end):
        nonlocal col
        for i in range(i_start, i_end):
            for j in range(j_start, j_end):
                for k in range(k_start, k_end):
                    if time.time() > deadline:
                        return
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
                    
    def add_strassen(i_start, j_start, k_start):
        nonlocal col
        if time.time() > deadline:
            return
        # M1
        U[i_start * m + j_start][col] = 1
        U[(i_start+1) * m + (j_start+1)][col] = 1
        V[j_start * p + k_start][col] = 1
        V[(j_start+1) * p + (k_start+1)][col] = 1
        W[k_start * n + i_start][col] = 1
        W[k_start * n + (i_start+1)][col] = 1
        W[(k_start+1) * n + i_start][col] = 1
        W[(k_start+1) * n + (i_start+1)][col] = 1
        col += 1
        # M2
        U[(i_start+1) * m + j_start][col] = 1
        U[(i_start+1) * m + (j_start+1)][col] = 1
        V[j_start * p + k_start][col] = 1
        W[k_start * n + (i_start+1)][col] = 1
        col += 1
        # M3
        U[i_start * m + j_start][col] = 1
        V[j_start * p + (k_start+1)][col] = 1
        V[(j_start+1) * p + (k_start+1)][col] = -1
        W[k_start * n + i_start][col] = 1
        W[(k_start+1) * n + i_start][col] = -1
        col += 1
        # M4
        U[(i_start+1) * m + (j_start+1)][col] = 1
        V[(j_start+1) * p + k_start][col] = 1
        V[j_start * p + k_start][col] = -1
        W[k_start * n + (i_start+1)][col] = 1
        W[(k_start+1) * n + (i_start+1)][col] = -1
        col += 1
        # M5
        U[i_start * m + j_start][col] = 1
        U[i_start * m + (j_start+1)][col] = 1
        V[(j_start+1) * p + (k_start+1)][col] = 1
        W[k_start * n + i_start][col] = 1
        W[(k_start+1) * n + i_start][col] = 1
        col += 1
        # M6
        U[(i_start+1) * m + j_start][col] = 1
        U[i_start * m + j_start][col] = -1
        V[j_start * p + k_start][col] = 1
        V[j_start * p + (k_start+1)][col] = 1
        W[k_start * n + (i_start+1)][col] = 1
        W[(k_start+1) * n + (i_start+1)][col] = 1
        col += 1
        # M7
        U[i_start * m + (j_start+1)][col] = 1
        U[(i_start+1) * m + (j_start+1)][col] = -1
        V[(j_start+1) * p + k_start][col] = 1
        V[(j_start+1) * p + (k_start+1)][col] = 1
        W[k_start * n + i_start][col] = 1
        W[(k_start+1) * n + i_start][col] = 1
        col += 1

    add_schoolbook(0, 1, 0, 4, 0, 2)
    add_schoolbook(0, 1, 0, 4, 2, 5)
    add_strassen(1, 0, 0)
    add_strassen(1, 2, 0)
    add_schoolbook(1, 3, 0, 2, 2, 5)
    add_schoolbook(1, 3, 2, 4, 2, 5)
    
    return {"U": U, "V": V, "W": W}
