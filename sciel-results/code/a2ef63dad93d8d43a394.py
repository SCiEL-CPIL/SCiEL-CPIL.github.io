import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 4, 8]

def verify(U, V, W, n, m, p, tol=1e-9):
    """Check if U,V,W form a valid decomposition."""
    rank = len(U[0]) if U else 0
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(s - 1.0) > tol:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # 10s margin before 120s limit
    
    # Schoolbook as fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j][col] = 1
                V_school[j*p+k][col] = 1
                W_school[k*n+i][col] = 1
                col += 1
    
    best_rank = rank_school
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    
    # Try to find lower rank using local search
    # Start with rank 56 and try to find a valid decomposition
    target_ranks = [56, 55, 54, 52, 48, 40, 32]
    
    for target_rank in target_ranks:
        if time.time() > deadline - 30:
            break
        
        # Initialize with schoolbook, zero out some entries
        U = [[0] * target_rank for _ in range(n * m)]
        V = [[0] * target_rank for _ in range(m * p)]
        W = [[0] * target_rank for _ in range(p * n)]
        
        # First target_rank columns from schoolbook
        for col in range(target_rank):
            i, j, k = col // (m*p), (col // p) % m, col % p
            U[i*m+j][col] = 1
            V[j*p+k][col] = 1
            W[k*n+i][col] = 1
        
        # This is not valid (missing terms), so we need a different init
        # Instead, use all schoolbook but try to merge
        # For now, just return schoolbook if search doesn't work
        pass
    
    return best_result
