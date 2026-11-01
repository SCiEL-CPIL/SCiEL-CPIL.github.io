import time
import numpy as np

DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin
    
    n, m, p = DIMENSIONS
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for t in range(6):
        for s in range(7):
            if time.time() > deadline:
                break
            r = 7 * t + s
            if s == 0:
                U[0][r] += 1
                U[3][r] += 1
                V[2*t][r] += 1
                V[12+2*t+1][r] += 1
                W[4*t][r] += 1
                W[4*t+3][r] += 1
            elif s == 1:
                U[2][r] += 1
                U[3][r] += 1
                V[2*t][r] += 1
                W[4*t+1][r] += 1
                W[4*t+3][r] += 1
            elif s == 2:
                U[0][r] += 1
                V[2*t+1][r] += 1
                V[12+2*t+1][r] -= 1
                W[4*t+2][r] += 1
                W[4*t+3][r] += 1
            elif s == 3:
                U[3][r] += 1
                V[12+2*t][r] += 1
                V[2*t][r] -= 1
                W[4*t][r] += 1
                W[4*t+1][r] += 1
            elif s == 4:
                U[0][r] += 1
                U[1][r] += 1
                V[12+2*t+1][r] += 1
                W[4*t+2][r] += 1
                W[4*t][r] -= 1
            elif s == 5:
                U[2][r] += 1
                U[0][r] -= 1
                V[2*t][r] += 1
                V[2*t+1][r] += 1
                W[4*t+3][r] += 1
            elif s == 6:
                U[1][r] += 1
                U[3][r] -= 1
                V[12+2*t][r] += 1
                V[12+2*t+1][r] += 1
                W[4*t][r] += 1
            if time.time() > deadline:
                break

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
