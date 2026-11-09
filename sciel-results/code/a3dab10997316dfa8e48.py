import time

DIMENSIONS = [3, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    rank = m * p
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for c in range(rank):
        jc = c // p
        kc = c % p
        
        # U[i*m + j][c] = 1 if j == jc
        for i in range(n):
            U[i * m + jc][c] = 1
            
        # V[j'*p + k][c] = 1 if j' == jc and k == kc
        V[jc * p + kc][c] = 1
        
        # W[k'*n + i][c] = 1 if k' == kc
        for i in range(n):
            W[kc * n + i][c] = 1
            
    return {"U": U, "V": V, "W": W}
