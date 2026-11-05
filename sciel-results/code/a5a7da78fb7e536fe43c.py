import time

DIMENSIONS = [3, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    # 6 disjoint Strassen blocks + schoolbook for remaining
    # Block 1: i{0,1} j{0,1} k{0,1}
    # Block 2: i{0,1} j{2,3} k{2,3}
    # Block 3: i{0,1} j{0,1} k{2,3}
    # Block 4: i{0,1} j{2,3} k{0,1}
    # Block 5: i{0,1} j{0,1} k{4,5}
    # Block 6: i{0,1} j{2,3} k{4,5}
    # Remaining: i=2 (28 terms) + i{0,1} j{0,1} k{6} (2) + i{0,1} j{2,3} k{6} (2) = 32
    rank = 6 * 7 + 32  # 42 + 32 = 74... wait let me recount
    
    # Actually let me recount covered terms:
    # 6 blocks * 8 terms = 48 covered
    # 84 - 48 = 36 remaining
    rank = 6 * 7 + 36  # 42 + 36 = 78
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_schoolbook(i, j, k, col):
        U[i * m + j][col] = 1
        V[j * p + k][col] = 1
        W[k * n + i][col] = 1

    def add_strassen(i0, i1, j0, j1, k0, k1, start_col):
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

    # 6 Strassen blocks
    add_strassen(0, 1, 0, 1, 0, 1, 0)    # Block 1
    add_strassen(0, 1, 2, 3, 2, 3, 7)    # Block 2
    add_strassen(0, 1, 0, 1, 2, 3, 14)   # Block 3
    add_strassen(0, 1, 2, 3, 0, 1, 21)   # Block 4
    add_strassen(0, 1, 0, 1, 4, 5, 28)   # Block 5
    add_strassen(0, 1, 2, 3, 4, 5, 35)   # Block 6
    
    # Schoolbook for remaining 36 terms
    covered = set()
    for i0, i1, j0, j1, k0, k1 in [
        (0, 1, 0, 1, 0, 1),
        (0, 1, 2, 3, 2, 3),
        (0, 1, 0, 1, 2, 3),
        (0, 1, 2, 3, 0, 1),
        (0, 1, 0, 1, 4, 5),
        (0, 1, 2, 3, 4, 5),
    ]:
        for ii in (i0, i1):
            for jj in (j0, j1):
                for kk in (k0, k1):
                    covered.add((ii, jj, kk))
    
    col = 42
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered:
                    add_schoolbook(i, j, k, col)
                    col += 1
                
    return {"U": U, "V": V, "W": W}
