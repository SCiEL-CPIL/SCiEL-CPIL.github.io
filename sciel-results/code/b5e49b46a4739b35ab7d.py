import numpy as np
import time

DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = DIMENSIONS  # n=3, m=3, p=5
    
    # We'll use a block decomposition approach:
    # Split B into B_left (3x3) and B_right (3x2)
    # Part 1: A * B_left (3x3 * 3x3 -> 3x3) - use Strassen for 2x2 block + schoolbook
    # Part 2: A * B_right (3x3 * 3x2 -> 3x2) - schoolbook
    
    # For Part 1 (3x3 * 3x3):
    # Split A and B_left into 2x2 blocks:
    # A = [[A11, A12], [A21, A22]] where A11 is 2x2, A12 is 2x1, A21 is 1x2, A22 is 1x1
    # B_left = [[B11, B12], [B21, B22]] similarly
    # 
    # For the 2x2 * 2x2 -> 2x2 part (A11 * B11), use Strassen (rank 7)
    # For the remaining terms, use schoolbook
    #
    # Actually, let me think more carefully. The tensor for 3x3*3x3 has rank at most 27 (schoolbook).
    # With Strassen on the top-left 2x2 block, we can potentially reduce it.
    #
    # Let me just implement a careful schoolbook decomposition first, then try to optimize.
    #
    # For 3x3 * 3x5:
    # U has shape (9, rank), V has shape (15, rank), W has shape (15, rank)
    # Index conventions:
    # U[i*m+j, r] for A[i,j]
    # V[j*p+k, r] for B[j,k]  
    # W[k*n+i, r] for C[k,i] (trace convention: W[k*n+i, r])
    
    # Schoolbook: rank = 3*3*5 = 45
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
