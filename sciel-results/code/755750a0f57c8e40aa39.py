import numpy as np
import time
import random

DIMENSIONS = [3, 3, 8]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Trivial decomposition as fallback
    rank_trivial = n * m * p
    U_trivial = [[0] * rank_trivial for _ in range(n * m)]
    V_trivial = [[0] * rank_trivial for _ in range(m * p)]
    W_trivial = [[0] * rank_trivial for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_trivial[i * m + j][col] = 1
                V_trivial[j * p + k][col] = 1
                W_trivial[k * n + i][col] = 1
                col += 1
    
    # Try to find a lower-rank decomposition
    # The tensor T[i,j,k] = 1 if the term A[i,j]*B[j,k] contributes to C[i,k]
    # We need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = T[i,j,k]
    
    # Try ranks from lower values up
    # Start with a reasonable upper bound and try to reduce
    # For 3x3x8, known best is likely close to 72, but let's try
    
    best_rank = rank_trivial
    best_U, best_V, best_W = U_trivial, V_trivial, W_trivial
    
    # Try a few candidate ranks
    for candidate_rank in [64, 60, 56, 52, 48, 44, 40, 36, 32, 28, 24]:
        if time.time() > deadline - 30:
            break
        
        # Try to solve for this rank using random initialization
        success = False
        for attempt in range(5):
            if time.time() > deadline - 30:
                break
            
            np.random.seed(attempt * 1000 + candidate_rank)
            
            U = np.random.randn(n * m, candidate_rank)
            V = np.random.randn(m * p, candidate_rank)
            W = np.random.randn(p * n, candidate_rank)
            
            # Build the tensor from the decomposition
            # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            T_decomp = np.zeros((n, m, p))
            for r in range(candidate_rank):
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            T_decomp[i, j, k] += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            
            # Target tensor: T[i,j,k] = 1 for all (i,j,k) where the term exists
            # Actually, for matrix multiplication, every (i,j,k) triple contributes
            T_target = np.ones((n, m, p))
            
            error = np.linalg.norm(T_decomp - T_target) / np.linalg.norm(T_target)
            
            if error < 1e-6:
                success = True
                break
        
        if success:
            best_rank = candidate_rank
            best_U = [[int(round(U[i, r])) for r in range(candidate_rank)] for i in range(n*m)]
            best_V = [[int(round(V[i, r])) for r in range(candidate_rank)] for i in range(m*p)]
            best_W = [[int(round(W[i, r])) for r in range(candidate_rank)] for i in range(p*n)]
            break
    
    return {"U": best_U, "V": best_V, "W": best_W}
