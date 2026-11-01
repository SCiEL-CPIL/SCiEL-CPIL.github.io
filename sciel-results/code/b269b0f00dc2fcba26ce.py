import numpy as np
import time

DIMENSIONS = [2, 2, 11]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 39 decomposition
    # 5 blocks of 2x2 (Strassen, rank 7 each)
    # 1 block of 2x1 (Schoolbook, rank 4)
    rank = 5 * 7 + 4
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    r = 0
    # 5 Strassen blocks
    for b in range(5):
        # B cols: 2*b, 2*b+1
        # V indices for B block:
        # v00 = 0*11 + 2*b = 2*b
        # v01 = 0*11 + 2*b + 1 = 2*b + 1
        # v10 = 1*11 + 2*b = 11 + 2*b
        # v11 = 1*11 + 2*b + 1 = 12 + 2*b
        
        v00 = 2*b
        v01 = 2*b + 1
        v10 = 11 + 2*b
        v11 = 12 + 2*b
        
        # W indices for C block:
        # w00 = (2*b)*2 + 0 = 4*b
        # w01 = (2*b+1)*2 + 0 = 4*b + 2
        # w10 = (2*b)*2 + 1 = 4*b + 1
        # w11 = (2*b+1)*2 + 1 = 4*b + 3
        
        w00 = 4*b
        w01 = 4*b + 2
        w10 = 4*b + 1
        w11 = 4*b + 3
        
        # U indices for A:
        # u00 = 0, u01 = 1, u10 = 2, u11 = 3
        
        # P1: (a00+a11)(b00+b11)
        U[0][r] += 1; U[3][r] += 1
        V[v00][r] += 1; V[v11][r] += 1
        W[w00][r] += 1; W[w11][r] += 1
        r += 1
        
        # P2: (a10+a11)b00
        U[2][r] += 1; U[3][r] += 1
        V[v00][r] += 1
        W[w10][r] += 1; W[w11][r] -= 1
        r += 1
        
        # P3: a00(b01-b11)
        U[0][r] += 1
        V[v01][r] += 1; V[v11][r] -= 1
        W[w01][r] += 1; W[w11][r] += 1
        r += 1
        
        # P4: a11(b10-b00)
        U[3][r] += 1
        V[v10][r] += 1; V[v00][r] -= 1
        W[w00][r] += 1; W[w10][r] += 1
        r += 1
        
        # P5: (a00+a01)b11
        U[0][r] += 1; U[1][r] += 1
        V[v11][r] += 1
        W[w00][r] -= 1; W[w01][r] += 1
        r += 1
        
        # P6: (a10-a00)(b00+b01)
        U[2][r] += 1; U[0][r] -= 1
        V[v00][r] += 1; V[v01][r] += 1
        W[w11][r] += 1
        r += 1
        
        # P7: (a01-a11)(b10+b11)
        U[1][r] += 1; U[3][r] -= 1
        V[v10][r] += 1; V[v11][r] += 1
        W[w00][r] += 1
        r += 1

    # 1 block of 2x1 at col 10
    # B[0, 10] -> V[10]
    # B[1, 10] -> V[21]
    # C[0, 10] -> W[20]
    # C[1, 10] -> W[21]
    
    # Q1: a00 * B[0,10] -> C[0,10]
    U[0][r] += 1
    V[10][r] += 1
    W[20][r] += 1
    r += 1
    
    # Q2: a01 * B[1,10] -> C[0,10]
    U[1][r] += 1
    V[21][r] += 1
    W[20][r] += 1
    r += 1
    
    # Q3: a10 * B[0,10] -> C[1,10]
    U[2][r] += 1
    V[10][r] += 1
    W[21][r] += 1
    r += 1
    
    # Q4: a11 * B[1,10] -> C[1,10]
    U[3][r] += 1
    V[21][r] += 1
    W[21][r] += 1
    r += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
