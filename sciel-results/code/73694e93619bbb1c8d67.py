# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    deadline = time.time() + 140
    n, m, p = 2, 2, 4
    
    # Rank 14 solution using Strassen's algorithm for two 2x2 blocks
    rank = 14
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Left half (k=0,1)
    # M1
    c = 0
    U[0][c] = 1; U[3][c] = 1
    V[0][c] = 1; V[5][c] = 1
    W[0][c] = 1; W[3][c] = 1
    # M2
    c = 1
    U[2][c] = 1; U[3][c] = 1
    V[0][c] = 1
    W[1][c] = 1; W[3][c] = -1
    # M3
    c = 2
    U[0][c] = 1
    V[1][c] = 1; V[5][c] = -1
    W[2][c] = 1; W[3][c] = 1
    # M4
    c = 3
    U[3][c] = 1
    V[4][c] = 1; V[0][c] = -1
    W[0][c] = 1; W[1][c] = 1
    # M5
    c = 4
    U[0][c] = 1; U[1][c] = 1
    V[5][c] = 1
    W[0][c] = -1; W[2][c] = 1
    # M6
    c = 5
    U[2][c] = 1; U[0][c] = -1
    V[0][c] = 1; V[1][c] = 1
    W[3][c] = 1
    # M7
    c = 6
    U[1][c] = 1; U[3][c] = -1
    V[4][c] = 1; V[5][c] = 1
    W[0][c] = 1
    
    # Right half (k=2,3)
    # M1
    c = 7
    U[0][c] = 1; U[3][c] = 1
    V[2][c] = 1; V[7][c] = 1
    W[4][c] = 1; W[7][c] = 1
    # M2
    c = 8
    U[2][c] = 1; U[3][c] = 1
    V[2][c] = 1
    W[5][c] = 1; W[7][c] = -1
    # M3
    c = 9
    U[0][c] = 1
    V[3][c] = 1; V[7][c] = -1
    W[6][c] = 1; W[7][c] = 1
    # M4
    c = 10
    U[3][c] = 1
    V[6][c] = 1; V[2][c] = -1
    W[4][c] = 1; W[5][c] = 1
    # M5
    c = 11
    U[0][c] = 1; U[1][c] = 1
    V[7][c] = 1
    W[4][c] = -1; W[6][c] = 1
    # M6
    c = 12
    U[2][c] = 1; U[0][c] = -1
    V[2][c] = 1; V[3][c] = 1
    W[7][c] = 1
    # M7
    c = 13
    U[1][c] = 1; U[3][c] = -1
    V[6][c] = 1; V[7][c] = 1
    W[4][c] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
