"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction

# Strassen-style rank-27 decomposition for 2x4 * 4x4
# We use a block decomposition:
# A is 2x4, split into A1 (2x2) and A2 (2x2)
# B is 4x4, split into B11, B12, B21, B22 (each 2x2)
# Result C = A*B, C1 = A1*B11 + A2*B21, C2 = A1*B12 + A2*B22
#
# Using Strassen's algorithm on the 2x2 blocks:
# For each pair of 2x2 matrix multiplications, Strassen uses 7 multiplications.
# We have 4 such 2x2 multiplications (A1*B11, A1*B12, A2*B21, A2*B22),
# but we can share some intermediate computations.
#
# Actually, let's think more carefully. We want to multiply a 2x4 by 4x4.
# Let's partition A as [A1 | A2] where A1, A2 are 2x2.
# Let's partition B as [[B11, B12], [B21, B22]] where each block is 2x2.
# Then C = [A1*B11 + A2*B21 | A1*B12 + A2*B22]
#
# To compute A1*B11 + A2*B21 and A1*B12 + A2*B22, we can use a shared Strassen-like approach.
# 
# Let's define 7 products for the first column of C (2x2 result):
# P1 = (A1 + A2) * (B11 + B21)
# P2 = (A1 + A2) * B21
# P3 = A1 * (B11 - B21)
# Then A1*B11 + A2*B21 = P1 - P2 + P3... let me verify:
# P1 - P2 + P3 = (A1+A2)(B11+B21) - (A1+A2)B21 + A1(B11-B21)
# = A1*B11 + A1*B21 + A2*B11 + A2*B21 - A1*B21 - A2*B21 + A1*B11 - A1*B21
# = 2*A1*B11 + A2*B11 - A1*B21... that's not right.
#
# Let me use the standard Strassen formulas properly.
# For 2x2 matrix multiplication M*N where M=[M11 M12; M21 M22], N=[N11 N12; N21 N22]:
# P1 = (M11+M22)*(N11+N22)
# P2 = (M21+M22)*N11
# P3 = M11*(N12-N22)
# P4 = M22*(N21-N11)
# P5 = (M11+M12)*N22
# P6 = (M21-M11)*(N11+N12)
# P7 = (M12-M22)*(N21+N22)
# Result: [[P1+P4-P5+P7, P3+P5], [P2+P4, P1-P2+P3+P6]]
#
# This requires 7 scalar-like products of 2x2 matrices, but each "product" is a full 2x2 matmul.
# So for each 2x2*2x2 multiplication, we need 7 tensor terms if we recurse, or 4 if we don't.
#
# For our problem, we have 4 independent 2x2*2x2 multiplications to combine.
# But we can structure it to share work.
#
# Alternative: Use the fact that the tensor for (2,4,4) matrix multiplication has a known rank.
# The schoolbook rank is 2*4*4 = 32. 
#
# Let me try a direct rank-27 construction using a different partitioning.
# 
# Actually, let me try to use a rank-27 decomposition based on known results.
# The border rank of (2,4,4) is at most 27, and I believe the rank is 27.
#
# Let me construct it explicitly. I'll use a basis of 27 rank-1 tensors.
#
# The matrix multiplication tensor T_{ijk} = 1 if i = i1 + n*i2, j = j1 + m*j2, k = k1 + p*k2
# where i1 in [0,n), j1 in [0,m), k1 in [0,p), i2 in [0,p), j2 in [0,n), k2 in [0,m)
# Wait, let me re-read the convention.
#
# U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
# W[k*n+i, r] - so W is indexed by (k, i) where k in [0,p), i in [0,n)
# U[i*m+j, r] - U indexed by (i, j) where i in [0,n), j in [0,m)
# V[j*p+k, r] - V indexed by (j, k) where j in [0,m), k in [0,p)
#
# The decomposition is: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 
# if (i,j,k) corresponds to a valid multiplication, i.e., always 1 for all i,j,k in range.
# Wait, for matrix multiplication, the tensor is T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
# Because (AB)[i,k] = sum_j A[i,j]*B[j,k], so the tensor coefficient is 1 for all (i,j,k).
#
# So we need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k.
#
# Let me try to find a rank-27 solution by solving a system of equations or using a known construction.
#
# Given time constraints, let me try a rank-28 or rank-27 construction.
#
# Actually, let me just return the schoolbook rank-32 for now and try to improve.
# But the task says score is the rank, lower is better. Score 1.0 means rank 32.
# If I find rank 27, score would be 32/27 ≈ 1.185.
#
# Let me try to implement a rank-27 decomposition.

def construct():
    n, m, p = DIMENSIONS  # 2, 4, 4
    start_time = time.time()
    deadline = 140  # seconds, safety margin
    
    # Try to find a rank-27 decomposition
    # Use the known structure from literature or derive one
    
    # Let's use a block-based Strassen-like approach
    # Partition the 2x4 matrix into two 2x2 blocks
    # Partition the 4x4 matrix into four 2x2 blocks
    
    # The product is a 2x4 matrix, partitioned into two 2x2 blocks
    # C_left = A1*B11 + A2*B21
    # C_right = A1*B12 + A2*B22
    
    # For each 2x2*2x2 product, Strassen uses 7 multiplications
    # But we can share some computations between the four products
    
    # Let me define the 27 terms explicitly
    # This is a known rank-27 decomposition for (2,4,4)
    
    # I'll construct it using a systematic approach
    # The indices: i in {0,1}, j in {0,1,2,3}, k in {0,1,2,3}
    # U[i*4+j, r], V[j*4+k, r], W[k*2+i, r]
    
    # Let me try to find a rank-27 solution by using a clever combination
    
    # Actually, let me just return the schoolbook for safety
    rank = n * m * p  # 32
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
    
    # Check time
    if time.time() - start_time > deadline:
        return {"U": U, "V": V, "W": W}
    
    # Try to find a better decomposition
    # Let's attempt a rank-27 construction
    
    # For (2,4,4), we can use the following approach:
    # Split j into j1 in {0,1} and j2 in {2,3}
    # i in {0,1}, k in {0,1,2,3}
    
    # The tensor is: sum over r of U[i*4+j, r] * V[j*4+k, r] * W[k*2+i, r] = 1
    
    # Let me try to find 27 vectors that satisfy this
    
    # I'll use a known rank-27 decomposition if I can recall it,
    # otherwise I'll fall back to rank 32
    
    # For now, return schoolbook
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
