import numpy as np
from fractions import Fraction
import time

DIMENSIONS = [2, 3, 3]

def _verify_decomposition(U, V, W, n=2, m=3, p=3):
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for r in range(rank):
                    val += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if val != 1:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    
    # Rank 17: Strassen (7) + schoolbook (10)
    rank = 17
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    # Strassen for 2x2x2 core
    # M1 = (A00+A11)(B00+B11)
    U[0][0] = 1; U[4][0] = 1
    V[0][0] = 1; V[4][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2 = (A10+A11)B00
    U[3][1] = 1; U[4][1] = 1
    V[0][1] = 1
    W[1][1] = 1
    
    # M3 = A00(B01-B11)
    U[0][2] = 1
    V[1][2] = 1; V[4][2] = -1
    W[2][2] = 1
    
    # M4 = A11(B10-B00)
    U[4][3] = 1
    V[3][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1
    
    # M5 = (A00+A01)B11
    U[0][4] = 1; U[1][4] = 1
    V[4][4] = 1
    W[0][4] = -1; W[2][4] = 1
    
    # M6 = (A10-A00)(B00+B01)
    U[3][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # M7 = (A01-A11)(B10+B11)
    U[1][6] = 1; U[4][6] = -1
    V[3][6] = 1; V[4][6] = 1
    W[0][6] = 1
    
    # Schoolbook for j=2 and k=2 terms
    r = 7
    for i in range(2):
        for k in range(3):
            U[i*3+2][r] = 1
            V[2*3+k][r] = 1
            W[k*2+i][r] = 1
            r += 1
    for i in range(2):
        for j in range(2):
            U[i*3+j][r] = 1
            V[j*3+2][r] = 1
            W[2*2+i][r] = 1
            r += 1
    
    if _verify_decomposition(U, V, W):
        return {"U": U, "V": V, "W": W}
    
    # Fallback
    rank = 18
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][r] = 1
                V[j*p+k][r] = 1
                W[k*n+i][r] = 1
                r += 1
    return {"U": U, "V": V, "W": W}
