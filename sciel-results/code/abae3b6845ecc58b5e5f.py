"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import sys

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    # Schoolbook decomposition of rank n*m*p
    start_time = time.time()
    DEADLINE = 120.0
    SAFETY_MARGIN = 5.0
    
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if time.time() - start_time > DEADLINE - SAFETY_MARGIN:
                    break
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
            if time.time() - start_time > DEADLINE - SAFETY_MARGIN:
                break
        if time.time() - start_time > DEADLINE - SAFETY_MARGIN:
            break
    
    # If we ran out of time mid-construction, return what we have
    # But schoolbook should finish instantly for n=2,m=3,p=10
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
