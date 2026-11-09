import numpy as np
import time

DIMENSIONS = [2, 3, 10]

def construct():
    n, m, p = DIMENSIONS
    
    # Baseline: Rank 55 using 5 blocks of 2x3x2 with rank 11 each
    total_rank = 55
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    for b in range(p // 2):
        c0 = 2 * b
        c1 = 2 * b + 1
        
        # Strassen 2x2x2 (Rank 7)
        # P1
        U[0][col] = 1; U[4][col] = 1
        V[c0][col] = 1; V[p+c1][col] = 1
        W[c0*n+0][col] = 1; W[c1*n+1][col] = 1
        col += 1
        # P2
        U[3][col] = 1; U[4][col] = 1
        V[c0][col] = 1
        W[c0*n+1][col] = 1; W[c1*n+1][col] = -1
        col += 1
        # P3
        U[0][col] = 1
        V[c1][col] = 1; V[p+c1][col] = -1
        W[c1*n+0][col] = 1; W[c1*n+1][col] = 1
        col += 1
        # P4
        U[4][col] = 1
        V[p+c0][col] = 1; V[c0][col] = -1
        W[c0*n+0][col] = 1; W[c0*n+1][col] = 1
        col += 1
        # P5
        U[0][col] = 1; U[1][col] = 1
        V[p+c1][col] = 1
        W[c0*n+0][col] = -1; W[c1*n+0][col] = 1
        col += 1
        # P6
        U[3][col] = 1; U[0][col] = -1
        V[c0][col] = 1; V[c1][col] = 1
        W[c1*n+1][col] = 1
        col += 1
        # P7
        U[1][col] = 1; U[4][col] = -1
        V[p+c0][col] = 1; V[p+c1][col] = 1
        W[c0*n+0][col] = 1
        col += 1

        # Standard 2x1x2 (Rank 4)
        U[2][col] = 1; V[2*p+c0][col] = 1; W[c0*n+0][col] = 1; col += 1
        U[2][col] = 1; V[2*p+c1][col] = 1; W[c1*n+0][col] = 1; col += 1
        U[5][col] = 1; V[2*p+c0][col] = 1; W[c0*n+1][col] = 1; col += 1
        U[5][col] = 1; V[2*p+c1][col] = 1; W[c1*n+1][col] = 1; col += 1

    return {"U": U, "V": V, "W": W}
