import numpy as np
import time

DIMENSIONS = [2, 2, 15]

def construct():
    n, m, p = DIMENSIONS  # n=2, m=2, p=15
    
    # We use a block decomposition.
    # 7 pairs of columns (0-1, 2-3, ..., 12-13) using Strassen (rank 7 each)
    # 1 single column (14) using trivial (rank 4)
    # Total rank = 7 * 7 + 4 = 53
    
    rank = 53
    # Using lists as in the seed program to ensure compatibility
    U = [[0.0] * rank for _ in range(n * m)]  # 4 x 53
    V = [[0.0] * rank for _ in range(m * p)]  # 30 x 53
    W = [[0.0] * rank for _ in range(p * n)]  # 30 x 53
    
    # A indices in U:
    # 0: A00, 1: A01, 2: A10, 3: A11
    
    # B indices in V:
    # j*p + k. j=0 -> 0..14, j=1 -> 15..29
    
    # C indices in W:
    # k*n + i. k=0 -> 0,1; k=1 -> 2,3; ... k=14 -> 28,29
    
    def add_strassen_block(base_r, k0, k1):
        # k0, k1 are the column indices in B (and C)
        r1 = base_r
        r2 = base_r + 1
        r3 = base_r + 2
        r4 = base_r + 3
        r5 = base_r + 4
        r6 = base_r + 5
        r7 = base_r + 6
        
        # M1 = (A00 + A11) * (B0k0 + B1k1)
        U[0][r1] += 1.0
        U[3][r1] += 1.0
        V[0*p + k0][r1] += 1.0
        V[1*p + k1][r1] += 1.0
        
        # M2 = (A10 + A11) * B0k0
        U[2][r2] += 1.0
        U[3][r2] += 1.0
        V[0*p + k0][r2] += 1.0
        
        # M3 = A00 * (B0k1 - B1k1)
        U[0][r3] += 1.0
        V[0*p + k1][r3] += 1.0
        V[1*p + k1][r3] += -1.0
        
        # M4 = A11 * (B1k0 - B0k0)
        U[3][r4] += 1.0
        V[1*p + k0][r4] += 1.0
        V[0*p + k0][r4] += -1.0
        
        # M5 = (A00 + A01) * B1k1
        U[0][r5] += 1.0
        U[1][r5] += 1.0
        V[1*p + k1][r5] += 1.0
        
        # M6 = (A10 - A00) * (B0k0 + B0k1)
        U[2][r6] += 1.0
        U[0][r6] += -1.0
        V[0*p + k0][r6] += 1.0
        V[0*p + k1][r6] += 1.0
        
        # M7 = (A01 - A11) * (B1k0 + B1k1)
        U[1][r7] += 1.0
        U[3][r7] += -1.0
        V[1*p + k0][r7] += 1.0
        V[1*p + k1][r7] += 1.0
        
        # Reconstruction W
        # C0k0 = M1 + M4 - M5 + M7
        W[2*k0 + 0][r1] += 1.0
        W[2*k0 + 0][r4] += 1.0
        W[2*k0 + 0][r5] += -1.0
        W[2*k0 + 0][r7] += 1.0
        
        # C1k0 = M2 + M4
        W[2*k0 + 1][r2] += 1.0
        W[2*k0 + 1][r4] += 1.0
        
        # C0k1 = M3 + M5
        W[2*k1 + 0][r3] += 1.0
        W[2*k1 + 0][r5] += 1.0
        
        # C1k1 = M1 - M2 + M3 + M6
        W[2*k1 + 1][r1] += 1.0
        W[2*k1 + 1][r2] += -1.0
        W[2*k1 + 1][r3] += 1.0
        W[2*k1 + 1][r6] += 1.0

    # Add 7 Strassen blocks for pairs (0,1), (2,3), ..., (12,13)
    for i in range(7):
        k0 = 2 * i
        k1 = 2 * i + 1
        base_r = 7 * i
        add_strassen_block(base_r, k0, k1)
        
    # Add trivial block for column 14
    base_r = 49
    k_last = 14
    
    r1 = base_r
    r2 = base_r + 1
    r3 = base_r + 2
    r4 = base_r + 3
    
    # Q1 = A00 * B0k_last
    U[0][r1] += 1.0
    V[0*p + k_last][r1] += 1.0
    W[2*k_last + 0][r1] += 1.0
    
    # Q2 = A01 * B1k_last
    U[1][r2] += 1.0
    V[1*p + k_last][r2] += 1.0
    W[2*k_last + 0][r2] += 1.0
    
    # Q3 = A10 * B0k_last
    U[2][r3] += 1.0
    V[0*p + k_last][r3] += 1.0
    W[2*k_last + 1][r3] += 1.0
    
    # Q4 = A11 * B1k_last
    U[3][r4] += 1.0
    V[1*p + k_last][r4] += 1.0
    W[2*k_last + 1][r4] += 1.0

    return {"U": U, "V": V, "W": W}
