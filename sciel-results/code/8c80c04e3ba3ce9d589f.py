"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 9]

import time
import numpy as np
from itertools import product

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 120
    
    # The tensor T has shape (n*m, m*p, p*n)
    # T[(i,j), (j2,k), (k2,i2)] = 1 if i==i2 and j==j2 and k==k2
    # We want to find rank-r decomposition:
    # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    
    # Schoolbook rank
    rank_schoolbook = n * m * p  # 54
    
    # Try to find a better decomposition
    # For this specific case, let's think about the structure
    
    # C[i,k] = sum_j A[i,j] B[j,k]
    # For each k, C[:,k] = A @ B[:,k]
    # This is 9 independent 2x3 x 3x1 multiplications
    
    # The tensor for 2x3 x 3x1: T[(i,j), k, l] = delta_{i,l} * delta_{j,k}
    # Let's try to find its rank
    
    # Actually, let's just try a direct search for the full tensor
    # Given time constraints, let's use the schoolbook as baseline and try to improve
    
    # Build the tensor
    N = n * m  # 6
    M = m * p  # 27
    P = p * n  # 18
    
    T = np.zeros((N, M, P), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Try rank r from some lower bound up to schoolbook
    # Lower bound: max of slice ranks
    # Slice along first mode: T[i*m+j, :, :] is M×P
    # The rank of the tensor is at least the max slice rank
    
    # Let's just try to find a decomposition with rank < 54
    # Use a simple approach: try rank 53, 52, ... and see if we can find one
    
    # Given the time limit, let's be strategic
    # The schoolbook is 54. Let's try to see if 53 works, etc.
    
    # Actually, for this problem, let me think about whether we can do better than 54.
    
    # One approach: use the fact that the 3x9 multiplication can be decomposed
    # in a clever way.
    
    # Let's just return the schoolbook decomposition for now, but try to improve
    # if we have time.
    
    # Schoolbook decomposition
    U = np.zeros((N, rank_schoolbook), dtype=int)
    V = np.zeros((M, rank_schoolbook), dtype=int)
    W = np.zeros((P, rank_schoolbook), dtype=int)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
