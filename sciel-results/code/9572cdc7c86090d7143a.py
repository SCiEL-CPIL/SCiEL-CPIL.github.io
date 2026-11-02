import time
import numpy as np

DIMENSIONS = [2, 3, 10]

def construct():
    start_time = time.time()
    deadline = start_time + 110

    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n

    # Rank 55 construction
    # Part 1: A1 (2x2) * B1 (2x10) -> Rank 35 (5 blocks of Strassen 2x2x2)
    # Part 2: A2 (2x1) * B2 (1x10) -> Rank 20 (Schoolbook)
    
    rank = 55
    U = np.zeros((nm, rank), dtype=float)
    V = np.zeros((mp, rank), dtype=float)
    W = np.zeros((pn, rank), dtype=float)

    col = 0

    # Part 1: 5 blocks of 2x2x2 Strassen
    for b in range(5):
        # B indices
        b00_idx = 2*b
        b01_idx = 2*b + 1
        b10_idx = 10 + 2*b
        b11_idx = 10 + 2*b + 1
        
        # A indices
        a00_idx = 0
        a01_idx = 1
        a10_idx = 3
        a11_idx = 4
        
        # C indices
        c00_idx = 4*b
        c01_idx = 4*b + 2
        c10_idx = 4*b + 1
        c11_idx = 4*b + 3
        
        # M1 = (A00+A11)(B00+B11)
        U[a00_idx, col] = 1.0
        U[a11_idx, col] = 1.0
        V[b00_idx, col] = 1.0
        V[b11_idx, col] = 1.0
        W[c00_idx, col] = 1.0
        W[c11_idx, col] = 1.0
        col += 1
        
        # M2 = (A10+A11)B00
        U[a10_idx, col] = 1.0
        U[a11_idx, col] = 1.0
        V[b00_idx, col] = 1.0
        W[c10_idx, col] = 1.0
        W[c11_idx, col] = -1.0
        col += 1
        
        # M3 = A00(B10-B11)
        U[a00_idx, col] = 1.0
        V[b10_idx, col] = 1.0
        V[b11_idx, col] = -1.0
        W[c01_idx, col] = 1.0
        W[c11_idx, col] = 1.0
        col += 1
        
        # M4 = A11(B01-B00)
        U[a11_idx, col] = 1.0
        V[b01_idx, col] = 1.0
        V[b00_idx, col] = -1.0
        W[c00_idx, col] = 1.0
        W[c10_idx, col] = 1.0
        col += 1
        
        # M5 = (A00+A01)B11
        U[a00_idx, col] = 1.0
        U[a01_idx, col] = 1.0
        V[b11_idx, col] = 1.0
        W[c00_idx, col] = -1.0
        W[c01_idx, col] = 1.0
        col += 1
        
        # M6 = (A10-A00)(B00+B01)
        U[a10_idx, col] = 1.0
        U[a00_idx, col] = -1.0
        V[b00_idx, col] = 1.0
        V[b01_idx, col] = 1.0
        W[c11_idx, col] = 1.0
        col += 1
        
        # M7 = (A01-A11)(B10+B11)
        U[a01_idx, col] = 1.0
        U[a11_idx, col] = -1.0
        V[b10_idx, col] = 1.0
        V[b11_idx, col] = 1.0
        W[c00_idx, col] = 1.0
        col += 1

    # Part 2: A2 (2x1) * B2 (1x10)
    for i in range(2):
        a_idx = i * 3 + 2
        for k in range(10):
            b_idx = 2 * 10 + k
            c_idx = k * 2 + i
            
            U[a_idx, col] = 1.0
            V[b_idx, col] = 1.0
            W[c_idx, col] = 1.0
            col += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
