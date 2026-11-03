import time

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p // 2 # 30? No, n*m*p is 60. Rank is 30.
    # Actually rank = m * p = 5 * 6 = 30.
    rank = m * p
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    r = 0
    for j in range(m):
        for k in range(p):
            # U indices: i=0, j and i=1, j
            U[0 * m + j][r] = 1
            U[1 * m + j][r] = 1
            
            # V index: j, k
            V[j * p + k][r] = 1
            
            # W indices: k, i=0 and k, i=1
            W[k * n + 0][r] = 1
            W[k * n + 1][r] = 1
            
            r += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
