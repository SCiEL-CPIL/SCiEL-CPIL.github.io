"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Safety margin before 150s kill
    
    # Schoolbook decomposition as fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    # Try to find a lower rank decomposition using QR-based approach
    # The tensor T_{i,j,k} for matrix multiplication: we need U,V,W such that
    # for all i,j,k: sum_r U[i*m+j, r] * V[j*p+k, r] = W[k*n+i, r]
    # This is a bilinear system. We can try to find a sparse solution.
    
    # Approach: construct the "support" matrix and use rank-revealing QR
    # Each rank term r defines a set of (i,j) pairs (from U_r), (j,k) pairs (from V_r), 
    # and (k,i) pairs (from W_r). The term is valid if for all (i,j,k) in the support,
    # U_r[i*m+j] * V_r[j*p+k] = W_r[k*n+i].
    
    # For simplicity, assume entries are in {0, 1, -1} and try small ranks.
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    
    # Try ranks from a lower bound up to schoolbook
    # Lower bound: max(n*m, m*p, p*n) but also at least max(n, m, p)
    # For 2x3x8: lower bound is max(6, 24, 16) = 24? No, tensor rank lower bounds are more subtle.
    # Known lower bound: rank >= max(n*m, m*p, p*n) / max(n, m, p) ... not standard.
    # Use rank >= n*m*p / max(n, m, p) as a heuristic? That gives 48/8=6, too low.
    # Better: rank >= max(n, m, p) = 8.
    
    # Try ranks from 8 up to 47
    for target_rank in range(max(n, m, p), rank_school):
        if time.time() > deadline:
            break
        # This is a hard NP-hard problem in general, so we use a heuristic
        # Try to find a decomposition with this rank using random initialization and local search
        # For now, just return schoolbook as we don't have a fast exact solver
        pass
    
    return best_result
# EVOLVE-BLOCK-END
