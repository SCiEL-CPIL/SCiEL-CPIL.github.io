import time

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    n, m, p = DIMENSIONS
    
    # Strategy: Decompose n=2 into two n=1 problems.
    # The tensor for (1, m, p) multiplication has rank m.
    # We can share the decomposition for both i=0 and i=1.
    # Rank R = m = 3.
    
    rank = m
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # U[i*m + j, r] = delta(j, r)
    for i in range(n):
        for j in range(m):
            U[i * m + j][j] = 1
            
    # V[j*p + k, r] = delta(j, r)
    for j in range(m):
        for k in range(p):
            V[j * p + k][j] = 1
            
    # W[k*n + i, r] = 1
    for k in range(p):
        for i in range(n):
            for r in range(rank):
                W[k * n + i][r] = 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
