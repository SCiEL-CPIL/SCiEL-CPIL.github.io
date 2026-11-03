"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 6]
import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s limit
    n, m, p = DIMENSIONS
    
    # First, create the schoolbook decomposition as a fallback
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school), dtype=float)
    V_school = np.zeros((m * p, rank_school), dtype=float)
    W_school = np.zeros((p * n, rank_school), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    best_rank = rank_school
    best_U = U_school
    best_V = V_school
    best_W = W_school
    
    # Try to find a lower-rank decomposition using random initialization and gradient-free search
    # We'll use a simple approach: try various rank values and see if we can find valid decompositions
    
    # The matrix multiplication tensor T has entries T[i,j,k] = 1 if and only if the (i,k) entry of AB
    # depends on A[i,j] and B[j,k]. The tensor is decomposed as sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Let's try to find a decomposition with lower rank using a greedy approach
    # We'll represent the tensor as a 3D array and try to find rank-1 approximations
    
    # Build the full tensor
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1.0  # All entries are 1 for standard matrix multiplication
    
    # Reshape for easier manipulation
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # The tensor equation is: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Try a simple approach: use the schoolbook but see if we can merge some terms
    # For now, return the schoolbook decomposition
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
