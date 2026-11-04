import numpy as np

DIMENSIONS = [2, 3, 10]

def construct():
    n, m, p = DIMENSIONS
    # We use 5 blocks of (2,3,2) with rank 11 each.
    # Total rank = 55.
    rank = 55
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for block in range(5):
        # Block handles columns 2*block+1 and 2*block+2 of B (1-indexed)
        # Which are indices 2*block and 2*block+1 (0-indexed)
        c1 = 2 * block
        c2 = 2 * block + 1
        
        # Strassen part for A[0:2, 0:2] * B[0:2, c1:c2+1]
        # A indices: (0,0), (0,1), (1,0), (1,1) -> U indices 0, 1, 3, 4
        # B indices: (0,c1), (0,c2), (1,c1), (1,c2) -> V indices 0*p+c1, 0*p+c2, 1*p+c1, 1*p+c2
        
        # M1 = A11 * (B11 + B22)
        U[0][col] = 1
        V[0 * p + c1][col] = 1
        V[1 * p + c2][col] = 1
        W[c1 * n + 0][col] = 1
        W[c2 * n + 1][col] = 1
        col += 1
        
        # M2 = (A11 + A21) * B22
        U[0][col] = 1
        U[3][col] = 1
        V[1 * p + c2][col] = 1
        W[c2 * n + 0][col] = 1
        W[c2 * n + 1][col] = 1
        col += 1
        
        # M3 = (A12 - A22) * (B11 + B21)
        U[1][col] = 1
        U[4][col] = -1
        V[0 * p + c1][col] = 1
        V[1 * p + c1][col] = 1
        W[c1 * n + 1][col] = 1
        W[c2 * n + 1][col] = 1
        col += 1
        
        # M4 = (A22 - A11) * (B12 + B22)
        U[4][col] = 1
        U[0][col] = -1
        V[0 * p + c2][col] = 1
        V[1 * p + c2][col] = 1
        W[c1 * n + 0][col] = 1
        W[c2 * n + 0][col] = 1
        col += 1
        
        # M5 = (A11 + A12) * B21
        U[0][col] = 1
        U[1][col] = 1
        V[1 * p + c1][col] = 1
        W[c1 * n + 0][col] = 1
        W[c2 * n + 0][col] = 1
        col += 1
        
        # M6 = (A21 + A22) * (B11 + B12)
        U[3][col] = 1
        U[4][col] = 1
        V[0 * p + c1][col] = 1
        V[0 * p + c2][col] = 1
        W[c1 * n + 1][col] = 1
        W[c2 * n + 1][col] = 1
        col += 1
        
        # M7 = (A11 - A22) * (B12 - B21)
        U[0][col] = 1
        U[4][col] = -1
        V[0 * p + c2][col] = 1
        V[1 * p + c1][col] = -1
        W[c1 * n + 0][col] = 1
        W[c2 * n + 1][col] = 1
        col += 1
        
        # Part 2: A[0:2, 2] * B[2, c1:c2+1]
        # A indices: (0,2), (1,2) -> U indices 2, 5
        # B indices: (2,c1), (2,c2) -> V indices 2*p+c1, 2*p+c2
        
        # Q1 = A13 * B31
        U[2][col] = 1
        V[2 * p + c1][col] = 1
        W[c1 * n + 0][col] = 1
        col += 1
        
        # Q2 = A13 * B32
        U[2][col] = 1
        V[2 * p + c2][col] = 1
        W[c2 * n + 0][col] = 1
        col += 1
        
        # Q3 = A23 * B31
        U[5][col] = 1
        V[2 * p + c1][col] = 1
        W[c1 * n + 1][col] = 1
        col += 1
        
        # Q4 = A23 * B32
        U[5][col] = 1
        V[2 * p + c2][col] = 1
        W[c2 * n + 1][col] = 1
        col += 1
        
    return {"U": U, "V": V, "W": W}
