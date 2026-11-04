import numpy as np
import time

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 66 decomposition
    rank = 66
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a Strassen 2x2x2 block
    def add_strassen(i1, i2, j1, j2, k1, k2):
        nonlocal col
        # M1 = (A11 + A22)(B11 + B22)
        # U: (i1,j1)=1, (i2,j2)=1
        # V: (j1,k1)=1, (j2,k2)=1
        # W: (k1,i1)=1, (k2,i2)=1
        U[i1 * m + j1][col] += 1
        U[i2 * m + j2][col] += 1
        V[j1 * p + k1][col] += 1
        V[j2 * p + k2][col] += 1
        W[k1 * n + i1][col] += 1
        W[k2 * n + i2][col] += 1
        col += 1
        
        # M2 = (A21 + A22) B11
        # U: (i2,j1)=1, (i2,j2)=1
        # V: (j1,k1)=1
        # W: (k1,i2)=1
        U[i2 * m + j1][col] += 1
        U[i2 * m + j2][col] += 1
        V[j1 * p + k1][col] += 1
        W[k1 * n + i2][col] += 1
        col += 1
        
        # M3 = A11 (B21 - B22)
        # U: (i1,j1)=1
        # V: (j2,k1)=1, (j2,k2)=-1
        # W: (k1,i1)=1
        U[i1 * m + j1][col] += 1
        V[j2 * p + k1][col] += 1
        V[j2 * p + k2][col] += -1
        W[k1 * n + i1][col] += 1
        col += 1
        
        # M4 = A22 (B12 - B11)
        # U: (i2,j2)=1
        # V: (j1,k2)=1, (j1,k1)=-1
        # W: (k2,i2)=1
        U[i2 * m + j2][col] += 1
        V[j1 * p + k2][col] += 1
        V[j1 * p + k1][col] += -1
        W[k2 * n + i2][col] += 1
        col += 1
        
        # M5 = (A11 + A12) B22
        # U: (i1,j1)=1, (i1,j2)=1
        # V: (j2,k2)=1
        # W: (k2,i1)=1
        U[i1 * m + j1][col] += 1
        U[i1 * m + j2][col] += 1
        V[j2 * p + k2][col] += 1
        W[k2 * n + i1][col] += 1
        col += 1
        
        # M6 = (A21 - A11) (B11 + B12)
        # U: (i2,j1)=1, (i1,j1)=-1
        # V: (j1,k1)=1, (j1,k2)=1
        # W: (k1,i2)=1, (k2,i2)=1
        U[i2 * m + j1][col] += 1
        U[i1 * m + j1][col] += -1
        V[j1 * p + k1][col] += 1
        V[j1 * p + k2][col] += 1
        W[k1 * n + i2][col] += 1
        W[k2 * n + i2][col] += 1
        col += 1
        
        # M7 = (A12 - A22) (B21 + B22)
        # U: (i1,j2)=1, (i2,j2)=-1
        # V: (j2,k1)=1, (j2,k2)=1
        # W: (k1,i1)=1, (k2,i1)=1
        U[i1 * m + j2][col] += 1
        U[i2 * m + j2][col] += -1
        V[j2 * p + k1][col] += 1
        V[j2 * p + k2][col] += 1
        W[k1 * n + i1][col] += 1
        W[k2 * n + i1][col] += 1
        col += 1

    # Part 1: C1 = A1 B (A1 is 2x4, B is 4x6)
    # A1 rows 0,1. B rows 0,1 (top) and 2,3 (bot).
    # A11 (cols 0,1) * B_top (rows 0,1)
    # B_top splits into 3 2x2 blocks for cols (0,1), (2,3), (4,5)
    
    # Block 1: A[0:2, 0:2] * B[0:2, 0:2]
    add_strassen(0, 1, 0, 1, 0, 1)
    # Block 2: A[0:2, 0:2] * B[0:2, 2:4]
    add_strassen(0, 1, 0, 1, 2, 3)
    # Block 3: A[0:2, 0:2] * B[0:2, 4:6]
    add_strassen(0, 1, 0, 1, 4, 5)
    
    # A12 (cols 2,3) * B_bot (rows 2,3)
    # B_bot splits into 3 2x2 blocks for cols (0,1), (2,3), (4,5)
    
    # Block 4: A[0:2, 2:4] * B[2:4, 0:2]
    add_strassen(0, 1, 2, 3, 0, 1)
    # Block 5: A[0:2, 2:4] * B[2:4, 2:4]
    add_strassen(0, 1, 2, 3, 2, 3)
    # Block 6: A[0:2, 2:4] * B[2:4, 4:6]
    add_strassen(0, 1, 2, 3, 4, 5)
    
    # Part 2: C2 = A2 B (A2 is 1x4, B is 4x6)
    # A2 row 2.
    # A21 (cols 0,1) * B_top (rows 0,1)
    for j in [0, 1]:
        for k in range(6):
            U[2 * m + j][col] += 1
            V[j * p + k][col] += 1
            W[k * n + 2][col] += 1
            col += 1
            
    # A22 (cols 2,3) * B_bot (rows 2,3)
    for j in [2, 3]:
        for k in range(6):
            U[2 * m + j][col] += 1
            V[j * p + k][col] += 1
            W[k * n + 2][col] += 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
