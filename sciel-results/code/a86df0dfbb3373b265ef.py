import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Known rank-17 decomposition
    # Core: 2x2x2 Strassen (7 terms)
    # Boundary: 10 terms (schoolbook)
    # Total: 17 terms
    
    rank = 17
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # Strassen terms for the 2x2x2 core (i in {0,1}, j in {0,1}, k in {0,1})
    # Indices:
    # U: i*3+j -> {0,1,3,4}
    # V: j*3+k -> {0,1,3,4}
    # W: k*2+i -> {0,1,2,3}
    
    # Term 0
    U[0][0] = 1; U[4][0] = 1
    V[0][0] = 1; V[4][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # Term 1
    U[3][1] = 1; U[4][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1
    
    # Term 2
    U[0][2] = 1
    V[1][2] = 1; V[4][2] = -1
    W[2][2] = 1; W[3][2] = 1
    
    # Term 3
    U[4][3] = 1
    V[3][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1
    
    # Term 4
    U[0][4] = 1; U[1][4] = 1
    V[4][4] = 1
    W[0][4] = -1; W[2][4] = 1
    
    # Term 5
    U[3][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # Term 6
    U[1][6] = 1; U[4][6] = -1
    V[3][6] = 1; V[4][6] = 1
    W[0][6] = 1
    
    # Boundary terms (j=2 or k=2)
    # (0,2,0): U[2], V[6], W[0]
    U[2][7] = 1; V[6][7] = 1; W[0][7] = 1
    # (0,2,1): U[2], V[7], W[2]
    U[2][8] = 1; V[7][8] = 1; W[2][8] = 1
    # (1,2,0): U[5], V[6], W[1]
    U[5][9] = 1; V[6][9] = 1; W[1][9] = 1
    # (1,2,1): U[5], V[7], W[3]
    U[5][10] = 1; V[7][10] = 1; W[3][10] = 1
    # (0,0,2): U[0], V[2], W[4]
    U[0][11] = 1; V[2][11] = 1; W[4][11] = 1
    # (0,1,2): U[1], V[5], W[4]
    U[1][12] = 1; V[5][12] = 1; W[4][12] = 1
    # (0,2,2): U[2], V[8], W[4]
    U[2][13] = 1; V[8][13] = 1; W[4][13] = 1
    # (1,0,2): U[3], V[2], W[5]
    U[3][14] = 1; V[2][14] = 1; W[5][14] = 1
    # (1,1,2): U[4], V[5], W[5]
    U[4][15] = 1; V[5][15] = 1; W[5][15] = 1
    # (1,2,2): U[5], V[8], W[5]
    U[5][16] = 1; V[8][16] = 1; W[5][16] = 1
    
    result = {"U": U, "V": V, "W": W}
    
    # Verification
    def verify(U, V, W):
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx_u = i * m + j
                    idx_v = j * p + k
                    idx_w = k * n + i
                    s = Fraction(0, 1)
                    for c in range(rank):
                        s += Fraction(U[idx_u][c]) * Fraction(V[idx_v][c]) * Fraction(W[idx_w][c])
                    if s != 1:
                        return False
        return True

    if not verify(U, V, W):
        # Fallback to schoolbook rank 18 if something went wrong (should not happen)
        rank = n * m * p
        U = [[0.0] * rank for _ in range(n * m)]
        V = [[0.0] * rank for _ in range(m * p)]
        W = [[0.0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1.0
                    V[j*p+k][col] = 1.0
                    W[k*n+i][col] = 1.0
                    col += 1
        return {"U": U, "V": V, "W": W}
        
    return result
