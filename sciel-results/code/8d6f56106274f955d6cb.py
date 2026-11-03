import time

DIMENSIONS = [2, 4, 5]
_START_TIME = time.time()
_DEADLINE = _START_TIME + 140

def construct():
    n, m, p = DIMENSIONS
    rank = 36
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen_222(i0, j0, k0, col_start):
        nonlocal col
        i0_idx = i0
        i1_idx = i0 + 1
        j0_idx = j0
        j1_idx = j0 + 1
        k0_idx = k0
        k1_idx = k0 + 1
        
        c = col_start
        # M1
        U[i0_idx * m + j0_idx][c] += 1
        U[i1_idx * m + j1_idx][c] += 1
        V[j0_idx * p + k0_idx][c] += 1
        V[j1_idx * p + k1_idx][c] += 1
        W[k0_idx * n + i0_idx][c] += 1
        W[k1_idx * n + i1_idx][c] += 1
        col += 1
        
        c = col_start + 1
        # M2
        U[i1_idx * m + j0_idx][c] += 1
        U[i1_idx * m + j1_idx][c] += 1
        V[j0_idx * p + k0_idx][c] += 1
        W[k0_idx * n + i1_idx][c] += 1
        W[k1_idx * n + i1_idx][c] -= 1
        col += 1
        
        c = col_start + 2
        # M3
        U[i0_idx * m + j0_idx][c] += 1
        V[j0_idx * p + k1_idx][c] += 1
        V[j1_idx * p + k1_idx][c] -= 1
        W[k1_idx * n + i0_idx][c] += 1
        W[k1_idx * n + i1_idx][c] += 1
        col += 1
        
        c = col_start + 3
        # M4
        U[i1_idx * m + j1_idx][c] += 1
        V[j1_idx * p + k0_idx][c] += 1
        V[j0_idx * p + k0_idx][c] -= 1
        W[k0_idx * n + i0_idx][c] += 1
        W[k0_idx * n + i1_idx][c] += 1
        col += 1
        
        c = col_start + 4
        # M5
        U[i0_idx * m + j0_idx][c] += 1
        U[i0_idx * m + j1_idx][c] += 1
        V[j1_idx * p + k1_idx][c] += 1
        W[k0_idx * n + i0_idx][c] -= 1
        W[k1_idx * n + i0_idx][c] += 1
        col += 1
        
        c = col_start + 5
        # M6
        U[i1_idx * m + j0_idx][c] += 1
        U[i0_idx * m + j0_idx][c] -= 1
        V[j0_idx * p + k0_idx][c] += 1
        V[j0_idx * p + k1_idx][c] += 1
        W[k1_idx * n + i1_idx][c] += 1
        col += 1
        
        c = col_start + 6
        # M7
        U[i0_idx * m + j1_idx][c] += 1
        U[i1_idx * m + j1_idx][c] -= 1
        V[j1_idx * p + k0_idx][c] += 1
        V[j1_idx * p + k1_idx][c] += 1
        W[k0_idx * n + i0_idx][c] += 1
        col += 1
        
        return col_start + 7

    def add_schoolbook_2x2x1(i0, j0, k0, col_start):
        c = col_start
        for di in range(2):
            for dj in range(2):
                U[(i0 + di) * m + (j0 + dj)][c] = 1
                V[(j0 + dj) * p + k0][c] = 1
                W[k0 * n + (i0 + di)][c] = 1
                c += 1
        return c

    # 1. C[:, 0:2] from A[:, 0:2] * B[0:2, 0:2]
    col = add_strassen_222(0, 0, 0, col)
    # 2. C[:, 0:2] from A[:, 2:4] * B[2:4, 0:2]
    col = add_strassen_222(0, 2, 0, col)
    # 3. C[:, 2:4] from A[:, 0:2] * B[0:2, 2:4]
    col = add_strassen_222(0, 0, 2, col)
    # 4. C[:, 2:4] from A[:, 2:4] * B[2:4, 2:4]
    col = add_strassen_222(0, 2, 2, col)
    # 5. C[:, 4] from A[:, 0:2] * B[0:2, 4]
    col = add_schoolbook_2x2x1(0, 0, 4, col)
    # 6. C[:, 4] from A[:, 2:4] * B[2:4, 4]
    col = add_schoolbook_2x2x1(0, 2, 4, col)
    
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    if time.time() > _DEADLINE:
        raise TimeoutError("Deadline exceeded")
    
    return {"U": U, "V": V, "W": W}
