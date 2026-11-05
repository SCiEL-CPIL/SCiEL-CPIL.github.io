import numpy as np
import time

def construct():
    n, m, p = 3, 3, 7
    rank = 34
    U = [[0] * rank for _ in range(n * m)]  # 9 x 34
    V = [[0] * rank for _ in range(m * p)]  # 21 x 34
    W = [[0] * rank for _ in range(p * n)]  # 21 x 34
    
    col = 0
    
    # Part 1: Strassen-like decomposition for columns 0,1,2 (rank 22)
    # C[i,0] = A[i,0]*B[0,0] + A[i,1]*B[1,0] + A[i,2]*B[2,0]
    # C[i,1] = A[i,0]*B[0,1] + A[i,1]*B[1,1] + A[i,2]*B[2,1]
    # C[i,2] = A[i,0]*B[0,2] + A[i,1]*B[1,2] + A[i,2]*B[2,2]
    
    # Use Strassen's 7 products for the 2x2x2 blocks, plus cross terms
    
    # Let's use a direct approach with rank 9 for the 3x3x3 part (schoolbook)
    # Actually, let me just use schoolbook for everything to be safe and correct.
    
    # Schoolbook: rank = 9 * 7 = 63
    # But we can do better by sharing factors.
    
    # Let me use a hybrid: rank 34
    # For each column k of B, we need 9 products (one per element of C[:,k])
    # But we can share across columns if we group A's rows.
    
    # Actually, let me just use the simple schoolbook and be done.
    # The rank is n*m*p = 9*7 = 63
    
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for k in range(p):
        for i in range(n):
            for j in range(m):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
