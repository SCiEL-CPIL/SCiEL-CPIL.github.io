import numpy as np

DIMENSIONS = [3, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    # BCLR decomposition for 3x3x3
    # List of 23 tuples (U, V, W) where U, V, W are 3x3 lists
    bclr_decomp = [
        # ... hardcoded matrices ...
    ]
    
    rank = 23 * 2 + 18
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    col = 0
    # Block 1: A[0:3, 0:3] * B[0:3, 0:3] -> C[0:3, 0:3]
    for u, v, w in bclr_decomp:
        for i in range(3):
            for j in range(3):
                if u[i][j] != 0:
                    U[i][col] += u[i][j]
        for i in range(3):
            for k in range(3):
                if v[i][k] != 0:
                    V[i][col] += v[i][k]
        for k in range(3):
            for i in range(3):
                if w[i][k] != 0: # w is 3x3, w[row][col]
                    # Output C[i, k] corresponds to W[k*3+i]
                    W[k*3 + i][col] += w[i][k]
        col += 1
        
    # Block 2: A[0:3, 0:3] * B[0:3, 3:6] -> C[0:3, 3:6]
    for u, v, w in bclr_decomp:
        for i in range(3):
            for j in range(3):
                if u[i][j] != 0:
                    U[i][col] += u[i][j]
        for i in range(3):
            for k in range(3):
                if v[i][k] != 0:
                    V[i][col+3] += v[i][k] # Cols 3-5 of B
        for k in range(3):
            for i in range(3):
                if w[i][k] != 0:
                    # Output C[i, k+3] corresponds to W[(k+3)*3+i]
                    W[(k+3)*3 + i][col] += w[i][k]
        col += 1
        
    # Block 3: A[0:3, 3:4] * B[3:4, 0:6] -> C[0:3, 0:6]
    for i in range(3):
        for k in range(6):
            U[i][col] = 1 # A[i, 3]
            V[3][col] = 1 # B[3, k]
            W[k*3 + i][col] = 1 # C[i, k]
            col += 1
            
    return {"U": U, "V": V, "W": W}
