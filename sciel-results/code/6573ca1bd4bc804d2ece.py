"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s buffer before 120s limit
    
    n, m, p = DIMENSIONS  # n=2, m=3, p=9
    
    # Schoolbook decomposition
    # Rank = n * m * p
    rank = n * m * p
    
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # We assign each (i, j, k) tuple to a unique rank r
    # r = i * (m * p) + j * p + k
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                r = i * (m * p) + j * p + k
                
                # U index: i * m + j
                U[i * m + j][r] = 1
                
                # V index: j * p + k
                V[j * p + k][r] = 1
                
                # W index: k * n + i
                W[k * n + i][r] = 1
                
                if time.time() > deadline:
                    break
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
    
    # Verify if we finished in time, otherwise return what we have (which might be invalid if incomplete)
    # But schoolbook is fast, so this should always complete.
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
