# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 3, 9
    rank = 18
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for q in range(3):
        for i in range(n):
            for j in range(m):
                for r in range(3):
                    a = i * m + j
                    b = j * p + (3 * q + r)
                    c = (3 * q + r) * n + i
                    U[a][col] = 1
                    V[b][col] = 1
                    W[c][col] = 1
                    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
