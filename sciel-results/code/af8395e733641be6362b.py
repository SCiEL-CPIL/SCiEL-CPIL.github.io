import time
import random
import numpy as np

DIMENSIONS = [2, 2, 3]

def construct():
    n, m, p = DIMENSIONS
    
    # Rank 11 decomposition for 2x2x3 matrix multiplication
    # Based on Strassen's algorithm for 2x2x2 extended to 2x2x3
    rank = 11
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # U indices: i*m + j
    # V indices: j*p + k
    # W indices: k*n + i
    
    # Term 1: M1 = (a00+a11)(b00+b11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[4][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # Term 2: M2 = (a10+a11)b00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1
    
    # Term 3: M3 = a00(b01-b11)
    U[0][2] = 1
    V[1][2] = 1; V[4][2] = -1
    W[2][2] = 1; W[3][2] = 1
    
    # Term 4: M4 = a11(b00-b01)
    U[3][3] = 1
    V[0][3] = 1; V[1][3] = -1
    W[0][3] = 1; W[1][3] = 1
    
    # Term 5: M5 = (a00+a01)b11
    U[0][4] = 1; U[1][4] = 1
    V[4][4] = 1
    W[0][4] = -1; W[2][4] = 1
    
    # Term 6: M6 = (a10-a00)(b00+b01)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # Term 7: M7 = (a01-a11)(b10+b11)
    U[1][6] = 1; U[3][6] = -1
    V[3][6] = 1; V[4][6] = 1
    W[0][6] = 1
    
    # Term 8: M8 = a00*b02
    U[0][7] = 1
    V[2][7] = 1
    W[4][7] = 1
    
    # Term 9: M9 = a01*b12
    U[1][8] = 1
    V[5][8] = 1
    W[4][8] = 1
    
    # Term 10: M10 = a10*b02
    U[2][9] = 1
    V[2][9] = 1
    W[5][9] = 1
    
    # Term 11: M11 = a11*b12
    U[3][10] = 1
    V[5][10] = 1
    W[5][10] = 1
    
    return {"U": U, "V": V, "W": W}
