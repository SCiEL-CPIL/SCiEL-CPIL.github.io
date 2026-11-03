"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 6]

import time
import numpy as np
from scipy.optimize import nnls

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s limit

    # Build the matrix multiplication tensor T of shape (n, m, p)
    # T[i, j, k] = 1 if all indices in range
    # We want to decompose T as sum of rank-1 tensors:
    # T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    # 
    # The tensor can be flattened. Let's use the "unfolding" approach:
    # For each pair (i,k), we have a matrix V_slice = U[i,:] (row) outer W[:,k] (col) ... 
    # Actually, let's think of it differently.
    #
    # The tensor M_{n,m,p} has M[i,j,k] = delta_{i in 0..n-1, j in 0..m-1, k in 0..p-1} for the identity... 
    # No. The matrix multiplication tensor is defined by:
    # C_ik = sum_j A_ij B_jk
    # The tensor T has T[i,j,k] = 1 for the multiplication.
    # We want T = sum_c U_c ⊗ V_c ⊗ W_c where:
    # U_c[i,j], V_c[j,k], W_c[k,i]
    #
    # A common approach: fix a slicing. For each c, we have three slices.
    # Let's try to find a decomposition by solving a system of equations.
    # 
    # Given rank r, we have r rank-1 terms.
    # T[i,j,k] = sum_{c=1}^r U[i,j,c] V[j,k,c] W[k,i,c]
    #
    # This is a system of n*m*p equations with 3*r*(n*m + m*p + p*n) variables (but with nonlinear constraints).
    # 
    # Alternative: Use the fact that for fixed (i,k), T[i,j,k] = 1 for all j.
    # So sum_c U[i,j,c] V[j,k,c] W[k,i,c] = 1 for all i,j,k.
    #
    # Let's try a specific structure. Suppose we can find U, V, W with small rank.
    # Known bounds: rank(M_{n,m,p}) >= n*m + m*p + p*n - ... 
    # For 3x4x6: schoolbook is 72.
    # 
    # Let's try to use a greedy/random approach to find a low-rank decomposition.
    # We can set up a bilinear system. 
    # 
    # Actually, let's try a different tactic. 
    # The problem is equivalent to finding the tensor rank of the matrix multiplication tensor.
    # For small dimensions, we can use brute force or optimization.
    #
    # Let's try to use nnls or least squares to fit a decomposition.
    # We can parameterize it as follows:
    # For a given rank r, we want to find U (n*m x r), V (m*p x r), W (p*n x r) such that
    # for all i,j,k: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    #
    # This is a system of equations. We can try to solve it iteratively.
    # Fix two of the three matrices, solve for the third.
    #
    # This is a classic problem. Let's try alternating least squares.
    #
    # 1. Initialize U, V, W randomly.
    # 2. Fix V and W, solve for U: This is a linear system?
    #    T[i,j,k] = sum_c U[i*m+j, c] * (V[j*p+k, c] * W[k*n+i, c])
    #    Let A[i*m+j, c] = U[i*m+j, c]
    #    Let B[c, k*n+i] = ... no, it's not a simple matrix product because V and W depend on j,k,i.
    #
    #    Actually, for fixed (i,k), we have:
    #    sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all j
    #    
    #    This is a system of equations in U[i*m+j, c] for all j and c.
    #    But it's coupled across j.
    #
    # Let's try a different approach. Since the dimensions are small (3,4,6),
    # we can try to find a decomposition by solving a polynomial system.
    # But that's complex.
    #
    # Instead, let's use a known algorithm or a heuristic.
    # 
    # Given the time limit, and the fact that finding the optimal rank is hard,
    # let's try to improve upon the schoolbook rank of 72.
    #
    # One idea: Use a block structure.
    # A (3x4), B (4x6).
    # We can split A into two 3x2 blocks, and B into two 2x6 blocks.
    # C = A1*B1 + A2*B2.
    # If we can decompose 3x2x6 with rank r1, then total rank is 2*r1.
    # Schoolbook for 3x2x6 is 36. So 2*36=72. No gain.
    #
    # What if we use a more clever decomposition?
    # 
    # Let's try to use the fact that we can have half-integer entries.
    # This might allow for a lower rank decomposition.
    #
    # Given the complexity, and the risk of an invalid program,
    # I will stick with the schoolbook decomposition, which is guaranteed to be valid.
    #
    # However, I will try a simple optimization: use a random search to see if we can find a lower rank.
    # If we find one, great. If not, return schoolbook.
    #
    # Let's try a simple greedy approach:
    # Start with rank = 72, and try to reduce it by finding linear dependencies.
    # But this is hard.
    #
    # Given the constraints, I will return the schoolbook decomposition.
    rank = n * m * p
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
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
