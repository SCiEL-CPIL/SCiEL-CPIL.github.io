import time

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 100
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for a in range(n):
        for b in range(m):
            for c in range(4):
                # u index: a * m + b
                u_idx = a * m + b
                # v index: ((b + c) % p) * p + ((a + c) % n)
                v_row = (b + c) % p
                v_col = (a + c) % n
                v_idx = v_row * p + v_col
                # w index: ((a + c) % p) * n + ((b + c) % n)
                w_row = (a + c) % p
                w_col = (b + c) % n
                w_idx = w_row * n + w_col
                
                U[u_idx][col] = 1
                V[v_idx][col] = 1
                W[w_idx][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
