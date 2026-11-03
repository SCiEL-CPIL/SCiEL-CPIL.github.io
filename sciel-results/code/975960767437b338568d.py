import time
import numpy as np

def construct():
    n, m, p = 2, 3, 8
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: A1 (2x1) * B1 (1x8) -> 2x8
    # A1 is A[:, 0]
    # B1 is B[0, :]
    # C_part1[i, k] = A[i, 0] * B[0, k]
    for i in range(2):
        for k in range(8):
            # U index: i*m + 0
            # V index: 0*p + k
            # W index: k*n + i
            U[i * m + 0][col] = 1
            V[0 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
            
    # Part 2: A2 (2x2) * B2 (2x8) -> 2x8
    # A2 is A[:, 1:3]
    # B2 is B[1:3, :]
    # Split B2 into 4 blocks of 2x2
    
    # Strassen decomposition for 2x2 * 2x2
    # A2 local indices:
    # a00 -> U[1], a01 -> U[2]
    # a10 -> U[4], a11 -> U[5]
    
    # B2 global indices for block b (cols 2b, 2b+1):
    # b00 -> V[1*p + 2b]
    # b01 -> V[1*p + 2b + 1]
    # b10 -> V[2*p + 2b]
    # b11 -> V[2*p + 2b + 1]
    
    # W indices for block b:
    # c00 -> W[2b*n + 0]
    # c01 -> W[(2b+1)*n + 0]
    # c10 -> W[2b*n + 1]
    # c11 -> W[(2b+1)*n + 1]
    
    for block in range(4):
        # 1. m1 = (a00+a11)(b00+b11)
        U[1][col] = 1
        U[5][col] = 1
        V[1 * p + 2 * block][col] = 1
        V[2 * p + 2 * block + 1][col] = 1
        W[2 * block * n + 0][col] = 1
        W[(2 * block + 1) * n + 1][col] = 1
        col += 1
        
        # 2. m2 = (a10+a11)b00
        U[4][col] = 1
        U[5][col] = 1
        V[1 * p + 2 * block][col] = 1
        W[2 * block * n + 1][col] = 1
        W[(2 * block + 1) * n + 1][col] = -1
        col += 1
        
        # 3. m3 = a00(b01-b11)
        U[1][col] = 1
        V[1 * p + 2 * block + 1][col] = 1
        V[2 * p + 2 * block + 1][col] = -1
        W[(2 * block + 1) * n + 0][col] = 1
        W[(2 * block + 1) * n + 1][col] = 1
        col += 1
        
        # 4. m4 = a11(b10-b00)
        U[5][col] = 1
        V[2 * p + 2 * block][col] = 1
        V[1 * p + 2 * block][col] = -1
        W[2 * block * n + 0][col] = 1
        W[2 * block * n + 1][col] = 1
        col += 1
        
        # 5. m5 = (a00+a01)b11
        U[1][col] = 1
        U[2][col] = 1
        V[2 * p + 2 * block + 1][col] = 1
        W[2 * block * n + 0][col] = -1
        W[(2 * block + 1) * n + 0][col] = 1
        col += 1
        
        # 6. m6 = (a10-a00)(b00+b01)
        U[4][col] = 1
        U[1][col] = -1
        V[1 * p + 2 * block][col] = 1
        V[1 * p + 2 * block + 1][col] = 1
        W[(2 * block + 1) * n + 1][col] = 1
        col += 1
        
        # 7. m7 = (a01-a11)(b10+b11)
        U[2][col] = 1
        U[5][col] = -1
        V[2 * p + 2 * block][col] = 1
        V[2 * p + 2 * block + 1][col] = 1
        W[2 * block * n + 0][col] = 1
        col += 1
        
    return {"U": U, "V": V, "W": W}
