import time
import numpy as np

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    r = 0
    # Part 1: j=2, all k (10 terms)
    for i in range(n):
        for k in range(p):
            U[i * m + 2][r] = 1
            V[2 * p + k][r] = 1
            W[k * n + i][r] = 1
            r += 1
    # Part 2: j=0,1; k=4 (4 terms)
    for i in range(n):
        for j in range(2):
            U[i * m + j][r] = 1
            V[j * p + 4][r] = 1
            W[4 * n + i][r] = 1
            r += 1
    # Part 3: j=0,1; k=0,1 Strassen (7 terms)
    U[0][r]=1; U[4][r]=1; V[0][r]=1; V[6][r]=1; W[0][r]=1; W[3][r]=1; r+=1
    U[1][r]=1; U[4][r]=1; V[0][r]=1; W[1][r]=1; W[3][r]=-1; r+=1
    U[0][r]=1; V[1][r]=1; V[6][r]=-1; W[2][r]=1; W[3][r]=1; r+=1
    U[4][r]=1; V[5][r]=1; V[0][r]=-1; W[0][r]=1; W[1][r]=1; r+=1
    U[0][r]=1; U[1][r]=1; V[6][r]=1; W[0][r]=-1; W[2][r]=1; r+=1
    U[1][r]=1; U[0][r]=-1; V[0][r]=1; V[1][r]=1; W[3][r]=1; r+=1
    U[1][r]=1; U[4][r]=-1; V[5][r]=1; V[6][r]=1; W[0][r]=1; r+=1
    # Part 4: j=0,1; k=2,3 Strassen (7 terms)
    U[0][r]=1; U[4][r]=1; V[2][r]=1; V[8][r]=1; W[4][r]=1; W[7][r]=1; r+=1
    U[1][r]=1; U[4][r]=1; V[2][r]=1; W[5][r]=1; W[7][r]=-1; r+=1
    U[0][r]=1; V[3][r]=1; V[8][r]=-1; W[6][r]=1; W[7][r]=1; r+=1
    U[4][r]=1; V[7][r]=1; V[2][r]=-1; W[4][r]=1; W[5][r]=1; r+=1
    U[0][r]=1; U[1][r]=1; V[8][r]=1; W[4][r]=-1; W[6][r]=1; r+=1
    U[1][r]=1; U[0][r]=-1; V[2][r]=1; V[3][r]=1; W[7][r]=1; r+=1
    U[1][r]=1; U[4][r]=-1; V[7][r]=1; V[8][r]=1; W[4][r]=1; r+=1
    
    return {"U": U, "V": V, "W": W}
