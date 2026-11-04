import numpy as np
import time

DIMENSIONS = [2, 3, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = DIMENSIONS
    # Rank 39 decomposition
    rank = 39
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    # Term 1: A_1 B_1
    # A_1 is col 0 of A (2x1). B_1 is row 0 of B (1x7).
    # i in 0..1, j=0, k in 0..6
    for i in range(n):
        for k in range(p):
            U[i * m + 0, col] = 1
            V[0 * p + k, col] = 1
            W[k * n + i, col] = 1
            col += 1
            
    # Term 2: A_2 B_2
    # A_2 is cols 1,2 of A (2x2). B_2 is rows 1,2 of B (2x7).
    # Local indices i, j in 0..1. Global j = j_local + 1.
    
    # Sub-terms for B_2 columns
    # B_2 is 2x7. Split into three 2x2 and one 2x1.
    # k ranges 0..6.
    # Block 1: k=0,1
    # Block 2: k=2,3
    # Block 3: k=4,5
    # Block 4: k=6
    
    def add_strassen_2x2x2(k_start):
        nonlocal col
        # Strassen coefficients for 2x2x2
        # M1 = A00(B00+B11)
        U[0 * m + 1, col] = 1
        V[1 * p + k_start + 0, col] = 1
        V[2 * p + k_start + 1, col] = 1
        W[(k_start + 0) * n + 0, col] = 1
        W[(k_start + 1) * n + 1, col] = 1
        col += 1
        
        # M2 = (A10+A11)(B00+B01)
        U[1 * m + 1, col] = 1
        U[1 * m + 2, col] = 1
        V[1 * p + k_start + 0, col] = 1
        V[1 * p + k_start + 1, col] = 1
        W[(k_start + 0) * n + 1, col] = 1
        W[(k_start + 1) * n + 1, col] = 1
        col += 1
        
        # M3 = (A00+A01)B11
        U[0 * m + 1, col] = 1
        U[0 * m + 2, col] = 1
        V[2 * p + k_start + 1, col] = 1
        W[(k_start + 1) * n + 0, col] = 1
        col += 1
        
        # M4 = A11(B10+B11)
        U[1 * m + 2, col] = 1
        V[2 * p + k_start + 0, col] = 1
        V[2 * p + k_start + 1, col] = 1
        W[(k_start + 0) * n + 1, col] = 1
        W[(k_start + 1) * n + 1, col] = 1
        col += 1
        
        # M5 = (A00-A10)(B00+B10)
        U[0 * m + 1, col] = 1
        U[1 * m + 1, col] = -1
        V[1 * p + k_start + 0, col] = 1
        V[2 * p + k_start + 0, col] = 1
        W[(k_start + 0) * n + 0, col] = -1
        W[(k_start + 1) * n + 1, col] = 1
        col += 1
        
        # M6 = (A01-A11)(B01+B11)
        U[0 * m + 2, col] = 1
        U[1 * m + 2, col] = -1
        V[1 * p + k_start + 1, col] = 1
        V[2 * p + k_start + 1, col] = 1
        W[(k_start + 1) * n + 0, col] = -1
        W[(k_start + 1) * n + 1, col] = 1
        col += 1
        
        # M7 = (A10+A00)B01
        U[1 * m + 1, col] = 1
        U[0 * m + 1, col] = 1
        V[1 * p + k_start + 1, col] = 1
        W[(k_start + 0) * n + 1, col] = -1
        W[(k_start + 1) * n + 0, col] = 1
        col += 1

    def add_2x2x1(k_start):
        nonlocal col
        # M1 = A00 B00
        U[0 * m + 1, col] = 1
        V[1 * p + k_start, col] = 1
        W[(k_start) * n + 0, col] = 1
        col += 1
        # M2 = A01 B10
        U[0 * m + 2, col] = 1
        V[2 * p + k_start, col] = 1
        W[(k_start) * n + 0, col] = 1
        col += 1
        # M3 = A10 B00
        U[1 * m + 1, col] = 1
        V[1 * p + k_start, col] = 1
        W[(k_start) * n + 1, col] = 1
        col += 1
        # M4 = A11 B10
        U[1 * m + 2, col] = 1
        V[2 * p + k_start, col] = 1
        W[(k_start) * n + 1, col] = 1
        col += 1

    add_strassen_2x2x2(0)
    add_strassen_2x2x2(2)
    add_strassen_2x2x2(4)
    add_2x2x1(6)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
