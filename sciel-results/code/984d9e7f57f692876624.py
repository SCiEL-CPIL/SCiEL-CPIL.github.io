import time

DIMENSIONS = [2, 3, 8]

def construct():
    n, m, p = DIMENSIONS
    num_groups = p // 2  # 4
    rank_per_group = 11
    rank = num_groups * rank_per_group  # 44
    
    U = [[0] * rank for _ in range(n * m)]   # (6, 44)
    V = [[0] * rank for _ in range(m * p)]   # (24, 44)
    W = [[0] * rank for _ in range(p * n)]   # (16, 44)
    
    for g in range(num_groups):
        base_col = g * rank_per_group
        skip = {(0, 0, 0), (1, 1, 1)}
        col = 0
        for i in range(n):
            for j in range(m):
                for k_local in range(2):
                    if (i, j, k_local) in skip:
                        continue
                    c = base_col + col
                    U[i * m + j][c] = 1
                    V[j * p + g * 2 + k_local][c] = 1
                    W[(g * 2 + k_local) * n + i][c] = 1
                    col += 1
        
        # Combined term
        c = base_col + col
        # (0,0,0): i=0, j=0, k_local=0
        U[0 * m + 0][c] = 1
        V[0 * p + g * 2 + 0][c] = 1
        W[(g * 2 + 0) * n + 0][c] = 1
        # (1,1,1): i=1, j=1, k_local=1
        U[1 * m + 1][c] = 1
        V[1 * p + g * 2 + 1][c] = 1
        W[(g * 2 + 1) * n + 1][c] = 1
    
    return {"U": U, "V": V, "W": W}
