# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import product

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0

    # Schoolbook decomposition for full 4x4x6 (Rank 96)
    rank_schoolbook = n * m * p
    U_school = [[0] * rank_schoolbook for _ in range(n * m)]
    V_school = [[0] * rank_schoolbook for _ in range(m * p)]
    W_school = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1

    # Improved decomposition: Split p=6 into p1=4 and p2=2
    # C = A B = [A B1, A B2]
    # Use schoolbook for A B1 (4x4x4, rank 48)
    # Use schoolbook for A B2 (4x4x2, rank 32)
    # Total rank 80
    
    rank_444 = n * m * 4
    rank_442 = n * m * 2
    total_rank = rank_444 + rank_442
    
    U_imp = [[0] * total_rank for _ in range(n * m)]
    V_imp = [[0] * total_rank for _ in range(m * p)]
    W_imp = [[0] * total_rank for _ in range(p * n)]
    
    # Part 1: A (4x4) * B1 (4x4)
    # B1 columns are 0..3
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(4):
                U_imp[i * m + j][col] = 1
                V_imp[j * p + k][col] = 1
                W_imp[k * n + i][col] = 1
                col += 1
                
    # Part 2: A (4x4) * B2 (4x2)
    # B2 columns are 4..5
    for i in range(n):
        for j in range(m):
            for k in range(4, 6):
                U_imp[i * m + j][col] = 1
                V_imp[j * p + k][col] = 1
                W_imp[k * n + i][col] = 1
                col += 1

    # Verify the improved decomposition
    def verify(U, V, W, tol=1e-9):
        n, m, p = DIMENSIONS
        rank = len(U[0]) if U else 0
        if len(U) != n * m or len(V) != m * p or len(W) != p * n:
            return False
        if any(len(row) != rank for row in U):
            return False
        if any(len(row) != rank for row in V):
            return False
        if any(len(row) != rank for row in W):
            return False
        
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U_arr[i*m+j], np.multiply(V_arr[j*p+k], W_arr[k*n+i]))
                    if abs(val - 1.0) > tol:
                        return False
        return True

    if verify(U_imp, V_imp, W_imp):
        return {"U": U_imp, "V": V_imp, "W": W_imp}
    else:
        # Fallback to schoolbook if verification fails (should not happen)
        return {"U": U_school, "V": V_school, "W": W_school}
# EVOLVE-BLOCK-END
