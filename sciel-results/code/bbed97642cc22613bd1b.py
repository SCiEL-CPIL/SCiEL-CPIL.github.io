"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 120  # 120s limit with safety margin
    
    n, m, p = DIMENSIONS  # 3, 5, 7
    rank = 5
    U = [[0] * rank for _ in range(n * m)]  # (15, 5)
    V = [[0] * rank for _ in range(m * p)]  # (35, 5)
    W = [[0] * rank for _ in range(p * n)]  # (21, 5)
    
    # Sub-problem 1: j=0,1 (rank 2)
    for i in range(n):
        U[i * m + 0][0] = 1
        U[i * m + 1][0] = 1
        U[i * m + 0][1] = 1
        U[i * m + 1][1] = -1
    for k in range(p):
        V[0 * p + k][0] = 1
        V[1 * p + k][0] = 1
        V[0 * p + k][1] = 1
        V[1 * p + k][1] = -1
    for i in range(n):
        for k in range(p):
            W[k * n + i][0] = 0.5
            W[k * n + i][1] = 0.5
    
    # Sub-problem 2: j=2,3 (rank 2)
    for i in range(n):
        U[i * m + 2][2] = 1
        U[i * m + 3][2] = 1
        U[i * m + 2][3] = 1
        U[i * m + 3][3] = -1
    for k in range(p):
        V[2 * p + k][2] = 1
        V[3 * p + k][2] = 1
        V[2 * p + k][3] = 1
        V[3 * p + k][3] = -1
    for i in range(n):
        for k in range(p):
            W[k * n + i][2] = 0.5
            W[k * n + i][3] = 0.5
    
    # Sub-problem 3: j=4 (rank 1)
    for i in range(n):
        U[i * m + 4][4] = 1
    for k in range(p):
        V[4 * p + k][4] = 1
    for i in range(n):
        for k in range(p):
            W[k * n + i][4] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
