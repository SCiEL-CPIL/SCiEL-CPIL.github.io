import time
import sys
import numpy as np

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 3, 5
    rank = 22
    U = [[0] * rank for _ in range(n * m)]  # 6 x 22
    V = [[0] * rank for _ in range(m * p)]  # 15 x 22
    W = [[0] * rank for _ in range(p * n)]  # 10 x 22
    
    r = 0
    
    # Part 1: A2 * B2 (rank 10)
    # A2 = A[:,2] (2x1), B2 = B[2,:] (1x5)
    for i in range(n):
        for k in range(p):
            U[i * m + 2][r] = 1
            V[2 * p + k][r] = 1
            W[k * n + i][r] = 1
            r += 1
    # r = 10
    
    # Part 2: A1 * B1 (rank 12)
    # A1 = A[:,:2] (2x2), B1 = B[:2,:] (2x5)
    # Use 12 terms
    
    # I'll use a construction based on:
    # For each pair of columns, use 4 terms, but share across 5 columns
    # Actually, let me use 12 explicit terms
    
    # Terms 11-12: handle column 0
    # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
    # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
    
    # Use 3 terms for column 0 (Strassen-like for 2x2x2x1)
    # Actually for 2x2 x 2x1, we need rank 7 normally, but we can do 3 if we allow sharing with other columns
    
    # Let me just use 12 terms total for A1*B1:
    # 4 terms for cols 0,1
    # 4 terms for cols 2,3  
    # 4 terms for col 4
    
    # Cols 0,1 (4 terms):
    # T11: (A[0,0]+A[0,1]) * (B[0,0]+B[1,0])
    U[0*m+0][r] = 1; U[0*m+1][r] = 1
    V[0*p+0][r] = 1; V[1*p+0][r] = 1
    W[0*n+0][r] = 1
    r += 1
    
    # T12: (A[0,0]+A[0,1]) * (B[0,1]+B[1,1])
    U[0*m+0][r] = 1; U[0*m+1][r] = 1
    V[0*p+1][r] = 1; V[1*p+1][r] = 1
    W[0*n+1][r] = 1
    r += 1
    
    # T13: (A[1,0]+A[1,1]) * (B[0,0]+B[1,0])
    U[1*m+0][r] = 1; U[1*m+1][r] = 1
    V[0*p+0][r] = 1; V[1*p+0][r] = 1
    W[1*n+0][r] = 1
    r += 1
    
    # T14: (A[1,0]+A[1,1]) * (B[0,1]+B[1,1])
    U[1*m+0][r] = 1; U[1*m+1][r] = 1
    V[0*p+1][r] = 1; V[1*p+1][r] = 1
    W[1*n+1][r] = 1
    r += 1
    
    # But this gives (A[0,0]+A[0,1])(B[0,0]+B[1,0]) = A[0,0]B[0,0] + A[0,0]B[1,0] + A[0,1]B[0,0] + A[0,1]B[1,0]
    # We only want A[0,0]B[0,0] + A[0,1]B[1,0]
    # So we need corrections. This approach uses too many terms.
    
    # Let me abandon this and just use the current rank-28 solution but try to optimize.
    # Actually, let me just return the current solution since I can't easily find rank 22.
    
    # Reset and use rank 28
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    r = 0
    
    # Part 1: A2 B2 (j=2, k=0..4) - rank 10
    for i in range(n):
        for k in range(p):
            U[i * m + 2][r] = 1
            V[2 * p + k][r] = 1
            W[k * n + i][r] = 1
            r += 1
            
    # Part 2: A1 B13 (j=0,1, k=4) - rank 4
    for i in range(n):
        for j in range(2):
            U[i * m + j][r] = 1
            V[j * p + 4][r] = 1
            W[4 * n + i][r] = 1
            r += 1
            
    # Part 3: A1 B11 (j=0,1, k=0,1) using Strassen - rank 7
    U[0 * m + 0][r] = 1; U[1 * m + 1][r] = 1
    V[0 * p + 0][r] = 1; V[1 * p + 1][r] = 1
    W[0 * n + 0][r] = 1; W[1 * n + 1][r] = 1
    r += 1
    U[1 * m + 0][r] = 1; U[1 * m + 1][r] = 1
    V[0 * p + 0][r] = 1
    W[0 * n + 1][r] = 1; W[1 * n + 1][r] = -1
    r += 1
    U[0 * m + 0][r] = 1
    V[0 * p + 1][r] = 1; V[1 * p + 1][r] = -1
    W[1 * n + 0][r] = 1; W[1 * n + 1][r] = 1
    r += 1
    U[1 * m + 1][r] = 1
    V[1 * p + 0][r] = 1; V[0 * p + 0][r] = -1
    W[0 * n + 0][r] = 1; W[0 * n + 1][r] = 1
    r += 1
    U[0 * m + 0][r] = 1; U[0 * m + 1][r] = 1
    V[1 * p + 1][r] = 1
    W[0 * n + 0][r] = -1; W[1 * n + 0][r] = 1
    r += 1
    U[1 * m + 0][r] = 1; U[0 * m + 0][r] = -1
    V[0 * p + 0][r] = 1; V[0 * p + 1][r] = 1
    W[1 * n + 1][r] = 1
    r += 1
    U[0 * m + 1][r] = 1; U[1 * m + 1][r] = -1
    V[1 * p + 0][r] = 1; V[1 * p + 1][r] = 1
    W[0 * n + 0][r] = 1
    r += 1
    
    # Part 4: A1 B12 (j=0,1, k=2,3) using Strassen - rank 7
    U[0 * m + 0][r] = 1; U[1 * m + 1][r] = 1
    V[0 * p + 2][r] = 1; V[1 * p + 3][r] = 1
    W[2 * n + 0][r] = 1; W[3 * n + 1][r] = 1
    r += 1
    U[1 * m + 0][r] = 1; U[1 * m + 1][r] = 1
    V[0 * p + 2][r] = 1
    W[2 * n + 1][r] = 1; W[3 * n + 1][r] = -1
    r += 1
    U[0 * m + 0][r] = 1
    V[0 * p + 3][r] = 1; V[1 * p + 3][r] = -1
    W[3 * n + 0][r] = 1; W[3 * n + 1][r] = 1
    r += 1
    U[1 * m + 1][r] = 1
    V[1 * p + 2][r] = 1; V[0 * p + 2][r] = -1
    W[2 * n + 0][r] = 1; W[2 * n + 1][r] = 1
    r += 1
    U[0 * m + 0][r] = 1; U[0 * m + 1][r] = 1
    V[1 * p + 3][r] = 1
    W[2 * n + 0][r] = -1; W[3 * n + 0][r] = 1
    r += 1
    U[1 * m + 0][r] = 1; U[0 * m + 0][r] = -1
    V[0 * p + 2][r] = 1; V[0 * p + 3][r] = 1
    W[3 * n + 1][r] = 1
    r += 1
    U[0 * m + 1][r] = 1; U[1 * m + 1][r] = -1
    V[1 * p + 2][r] = 1; V[1 * p + 3][r] = 1
    W[2 * n + 0][r] = 1
    r += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
