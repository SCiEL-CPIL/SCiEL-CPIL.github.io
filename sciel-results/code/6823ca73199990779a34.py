import numpy as np
import time

DIMENSIONS = [2, 4, 8]

def construct():
    """
    Rank-56 tensor decomposition for 2x4 * 4x8 matrix multiplication.
    Uses Strassen's algorithm on 2x2 blocks.
    """
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS
    rank = 56
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for s in range(2):
        for j in range(4):
            if time.time() > deadline:
                return {"U": U, "V": V, "W": W}
            
            # Indices for A (2x2 block)
            a00 = 0 * m + (2*s)
            a01 = 0 * m + (2*s + 1)
            a10 = 1 * m + (2*s)
            a11 = 1 * m + (2*s + 1)
            
            # Indices for B (2x2 block)
            b00 = (2*s) * p + (2*j)
            b01 = (2*s) * p + (2*j + 1)
            b10 = (2*s + 1) * p + (2*j)
            b11 = (2*s + 1) * p + (2*j + 1)
            
            # Indices for C (2x2 block)
            c00 = (2*j) * n + 0
            c01 = (2*j + 1) * n + 0
            c10 = (2*j) * n + 1
            c11 = (2*j + 1) * n + 1
            
            # Strassen coefficients
            # M1 = (A00 + A11) * (B00 + B11)
            U[a00][col] += 1
            U[a11][col] += 1
            V[b00][col] += 1
            V[b11][col] += 1
            W[c00][col] += 1
            W[c01][col] += 1
            W[c10][col] += 1
            W[c11][col] += 1
            
            # M2 = (A10 + A11) * B00
            U[a10][col+1] += 1
            U[a11][col+1] += 1
            V[b00][col+1] += 1
            W[c10][col+1] += 1
            W[c11][col+1] += 1
            
            # M3 = A00 * (B01 - B11)
            U[a00][col+2] += 1
            V[b01][col+2] += 1
            V[b11][col+2] += -1
            W[c00][col+2] += 1
            W[c01][col+2] += 1
            
            # M4 = A11 * (B10 - B00)
            U[a11][col+3] += 1
            V[b10][col+3] += 1
            V[b00][col+3] += -1
            W[c10][col+3] += 1
            W[c11][col+3] += 1
            
            # M5 = (A00 + A01) * B11
            U[a00][col+4] += 1
            U[a01][col+4] += 1
            V[b11][col+4] += 1
            W[c00][col+4] += -1
            W[c01][col+4] += 1
            W[c10][col+4] += -1
            W[c11][col+4] += 1
            
            # M6 = (A10 - A00) * (B00 + B01)
            U[a10][col+5] += 1
            U[a00][col+5] += -1
            V[b00][col+5] += 1
            V[b01][col+5] += 1
            W[c10][col+5] += 1
            W[c11][col+5] += 1
            
            # M7 = (A01 - A11) * (B10 + B11)
            U[a01][col+6] += 1
            U[a11][col+6] += -1
            V[b10][col+6] += 1
            V[b11][col+6] += 1
            W[c00][col+6] += -1
            W[c01][col+6] += 1
            W[c10][col+6] += -1
            W[c11][col+6] += 1
            
            col += 7
            
    return {"U": U, "V": V, "W": W}
