DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = m + p - 1
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r in range(rank):
        for i in range(n):
            for j in range(m):
                if i == 0:
                    if j <= r:
                        U[i * m + j][r] = 1
                else:
                    if r < j + p:
                        U[i * m + j][r] = 1
                        
            for j in range(m):
                for k in range(p):
                    if j <= r and k <= r - j:
                        V[j * p + k][r] = 1
                        
            for k in range(p):
                for i in range(n):
                    if i == 0:
                        if k <= r:
                            W[k * n + i][r] = 1
                    else:
                        if r < k + p:
                            W[k * n + i][r] = 1
                            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
