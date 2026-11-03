"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def construct():
    start_time = time.time()
    deadline = 135.0  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    # n=3, m=3, p=8
    
    # For matrix multiplication C = A @ B where A is n x m and B is m x p,
    # the tensor T[i,j,k] corresponds to A[i,j]*B[j,k] contributing to C[i,k].
    # 
    # In the decomposition framework:
    # U[i*m + j, t] = coefficient of A[i,j] in the t-th factor
    # V[j*p + k, t] = coefficient of B[j,k] in the t-th factor  
    # W[k*n + i, t] = coefficient for C[i,k] from the t-th factor
    #
    # Schoolbook: rank = n*m*p = 72
    # Each term: U[i*m+j, t]=1, V[j*p+k, t]=1, W[k*n+i, t]=1
    #
    # Strategy: Try to find a rank-deficient decomposition by exploiting
    # linear dependencies. The matrix multiplication tensor M_{n,m,p} can be
    # viewed as a 3-tensor of shape (n*m) x (m*p) x (p*n).
    #
    # For small cases, we can attempt to find a low-rank decomposition using
    # iterative methods or known algebraic constructions.
    #
    # Known results:
    # - M_{2,2,2} has rank 7 (Strassen) vs schoolbook 8
    # - M_{3,3,1} has rank 19 (best known) vs schoolbook 9... wait, that's not right
    # - Actually M_{3,3,3} has rank 23 (best known) vs schoolbook 27
    #
    # For M_{3,3,8}, we can think of it as computing 8 separate 3x3 matrix-vector
    # products, or as a 3x3 matrix times a 3x8 matrix.
    #
    # Let me try a systematic approach: use the fact that the tensor can be
    # sliced along different modes and look for rank reductions.
    #
    # Approach: Start with schoolbook rank 72, then try to merge terms.
    # Two schoolbook terms (i1,j1,k1) and (i2,j2,k2) can potentially be merged
    # if their contributions to C are linearly dependent in some way.
    #
    # Actually, let me try a different approach: use random initialization and
    # gradient-based optimization to find a low-rank decomposition.
    #
    # The tensor T has shape (9, 24, 24) since n*m=9, m*p=24, p*n=24.
    # We want to find U (9 x r), V (24 x r), W (24 x r) such that
    # T[i,j,k] = sum_t U[i,t] * V[j,t] * W[k,t]
    #
    # where i indexes A entries (i*m+j), j indexes B entries (j*p+k), 
    # k indexes C entries (k*n+i).
    #
    # Let me build the full tensor first.
    
    # Build the matrix multiplication tensor T of shape (n*m, m*p, p*n)
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m + j, j*p + k, k*n + i] = 1.0
    
    # Try to find a low-rank decomposition
    # Start with schoolbook rank 72
    rank_schoolbook = n * m * p
    
    # Try iterative rank reduction using ALS (Alternating Least Squares)
    # or similar methods. But first, let's see if we can find a known
    # algebraic construction.
    
    # For M_{3,3,8}, one approach is to use the fact that we can precompute
    # linear combinations of B's rows. For example, if we let:
    # d0 = B[0,:] + B[1,:]
    # d1 = B[1,:] + B[2,:]
    # d2 = B[0,:] - B[1,:]
    # etc.
    # Then some products can be expressed using these precomputed sums.
    
    # Let me try a specific construction:
    # For the 3x3 matrix A times 3x8 matrix B:
    # C[0,:] = A[0,0]*B[0,:] + A[0,1]*B[1,:] + A[0,2]*B[2,:]
    # C[1,:] = A[1,0]*B[0,:] + A[1,1]*B[1,:] + A[1,2]*B[2,:]
    # C[2,:] = A[2,0]*B[0,:] + A[2,1]*B[1,:] + A[2,2]*B[2,:]
    
    # Precompute:
    # s0 = B[0,:] + B[1,:]
    # s1 = B[1,:] + B[2,:]
    # s2 = B[0,:] - B[1,:]
    # s3 = B[1,:] - B[2,:]
    # s4 = B[0,:] + B[2,:]
    
    # Then:
    # A[0,0]*B[0,:] + A[0,1]*B[1,:] = (A[0,0]+A[0,1])/2 * s0 + (A[0,0]-A[0,1])/2 * s2
    # This uses 2 multiplications instead of 2, no savings.
    
    # To get savings, we need to share precomputed sums across multiple
    # output entries. For example:
    # C[0,:] + C[1,:] = (A[0,0]+A[1,0])*B[0,:] + (A[0,1]+A[1,1])*B[1,:] + (A[0,2]+A[1,2])*B[2,:]
    # This doesn't directly help.
    
    # Let me try a more systematic search: use a greedy algorithm to merge
    # schoolbook terms.
    
    # Actually, let me just try to find a rank-68 decomposition by removing
    # 4 terms and adjusting the remaining ones.
    
    # For now, return schoolbook as a safe baseline.
    rank = rank_schoolbook
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    result = {"U": U, "V": V, "W": W}
    return result
# EVOLVE-BLOCK-END
