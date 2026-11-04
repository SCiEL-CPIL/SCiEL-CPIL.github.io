import numpy as np
import time

DIMENSIONS = [2, 4, 8]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    n, m, p = DIMENSIONS
    # n=2, m=4, p=8
    # A is 2x4, B is 4x8, C is 2x8
    
    # Rank 56: 8 independent 2x2 * 2x2 Strassen decompositions
    rank = 56
    
    U = [[0] * rank for _ in range(n * m)]  # 8 x 56
    V = [[0] * rank for _ in range(m * p)]  # 32 x 56
    W = [[0] * rank for _ in range(p * n)]  # 16 x 56
    
    col = 0
    for s in range(2):  # s=0,1: split A columns / B rows into 2x2 blocks
        for j in range(4):  # j=0..3: split B columns into 2x2 blocks
            if time.time() > deadline:
                return {"U": U, "V": V, "W": W}
            
            # A_s: rows 0-1, cols 2s, 2s+1 of A (2x4)
            a00 = 0 * m + (2*s + 0)
            a01 = 0 * m + (2*s + 1)
            a10 = 1 * m + (2*s + 0)
            a11 = 1 * m + (2*s + 1)
            
            # B_{s,j}: rows 2s, 2s+1; cols 2j, 2j+1 of B (4x8)
            b00 = (2*s + 0) * p + (2*j + 0)
            b01 = (2*s + 0) * p + (2*j + 1)
            b10 = (2*s + 1) * p + (2*j + 0)
            b11 = (2*s + 1) * p + (2*j + 1)
            
            # C block: rows 0-1, cols 2j, 2j+1; global index = col*n + row
            c00 = (2*j + 0) * n + 0
            c01 = (2*j + 1) * n + 0
            c10 = (2*j + 0) * n + 1
            c11 = (2*j + 1) * n + 1
            
            # Strassen U (A-side) coefficients
            U[a00][col] += 1
            U[a11][col] += 1
            U[a10][col+1] += 1
            U[a11][col+1] += 1
            U[a00][col+2] += 1
            U[a11][col+3] += 1
            U[a00][col+4] += 1
            U[a01][col+4] += 1
            U[a10][col+5] += 1
            U[a00][col+5] += -1
            U[a01][col+6] += 1
            U[a11][col+6] += -1
            
            # Strassen V (B-side) coefficients
            V[b00][col] += 1
            V[b11][col] += 1
            V[b00][col+1] += 1
            V[b01][col+2] += 1
            V[b11][col+2] += -1
            V[b10][col+3] += 1
            V[b00][col+3] += -1
            V[b11][col+4] += 1
            V[b00][col+5] += 1
            V[b01][col+5] += 1
            V[b10][col+6] += 1
            V[b11][col+6] += 1
            
            # Strassen W (C-side) coefficients
            W[c00][col] += 1
            W[c00][col+3] += 1
            W[c00][col+4] += -1
            W[c00][col+6] += 1
            W[c01][col+2] += 1
            W[c01][col+4] += 1
            W[c10][col+1] += 1
            W[c10][col+3] += 1
            W[c11][col] += 1
            W[c11][col+1] += -1
            W[c11][col+2] += 1
            W[c11][col+5] += 1
            
            col += 7
    
    return {"U": U, "V": V, "W": W}
