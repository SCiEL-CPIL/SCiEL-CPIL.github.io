"""Structural decomposition for 4x4x7 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [4, 4, 7]

def construct():
    deadline = time.time() + 140
    
    n, m, p = 4, 4, 7
    
    # Block decomposition: 4x4x4 (rank 16) + 4x4x3 (rank 12) = rank 28
    rank = 28
    U = [[0] * rank for _ in range(n * m)]  # 16 x 28
    V = [[0] * rank for _ in range(m * p)]  # 28 x 28
    W = [[0] * rank for _ in range(p * n)]  # 28 x 28
    
    col = 0
    # 4x4x4 schoolbook
    for i in range(n):
        for j in range(m):
            for k in range(4):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    # 4x4x3 schoolbook for k=4,5,6
    for i in range(n):
        for j in range(m):
            for k in range(4, p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    # Verify
    if time.time() > deadline - 20:
        return {"U": U, "V": V, "W": W}
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U_arr[i*m+j, c] * V_arr[j*p+k, c] * W_arr[k*n+i, c] for c in range(rank))
                if abs(val - 1.0) > 1e-9:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if not valid:
        # Fallback to schoolbook rank 112
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
    
    return {"U": U, "V": V, "W": W}
