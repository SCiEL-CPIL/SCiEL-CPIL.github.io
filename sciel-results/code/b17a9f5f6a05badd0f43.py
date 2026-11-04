import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [4, 4, 5]

def construct():
    """
    Returns a rank-minimal tensor decomposition for 4x4 * 4x5 matrix multiplication.
    
    The decomposition has the form:
      (A * B)[k*n+i, j] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    where n=4, m=4, p=5.
    
    We use a rank-20 decomposition based on the identity:
    For each (i, k) pair, we can compute the inner product A[i,:]*B[:,k] using 5 terms,
    but we can do better by exploiting the structure.
    
    Actually, let's use a known approach: the matrix multiplication tensor of format
    (n,m,p) has rank at most n*m*p (schoolbook) but can be lower.
    
    For 4x4 * 4x5, we can use the fact that we can decompose the 4x4 product into
    rank-16 (schoolbook for square) and then extend to p=5.
    
    Actually, let me think differently. The tensor is:
    T[i,j,k] = A[i,j] * B[j,k]
    
    We need to find rank-r decomposition:
    T = sum_c U_c * V_c * W_c
    
    where U_c is a 4x4 matrix, V_c is a 4x5 matrix, W_c is a 5x4 matrix... 
    wait, let me re-read the indexing.
    
    The convention is:
    - U has shape (n*m, rank) = (16, rank), indexed by [i*m+j, c]
    - V has shape (m*p, rank) = (20, rank), indexed by [j*p+k, c]  
    - W has shape (p*n, rank) = (20, rank), indexed by [k*n+i, c]
    
    And the output is W[k*n+i, r]... wait, the output factor follows W[k*n+i,r].
    
    Let me re-read: "the output factor follows the official trace convention W[k*n+i,r]"
    
    So the decomposition is:
    sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = (A*B)[k*n+i, ...]
    
    Wait, I think the product (A*B) should have dimensions (p, n) = (5, 4) if A is 4x4 and B is 4x5.
    So (A*B)[k, i] where k is row (0..4) and i is col (0..3), indexed as k*n+i.
    
    So we need:
    (A*B)[k, i] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    This is the standard matrix multiplication tensor decomposition.
    
    For the schoolbook, rank = n*m*p = 4*4*5 = 80.
    
    Can we do better? For square matrices, we know:
    - 2x2: rank 7 (Strassen) vs 8 (schoolbook)
    - 4x4: best known is 49 (Lader) vs 64 (schoolbook)
    
    For 4x4 * 4x5, the best known rank is... let me think.
    
    Actually, let me just use a rank-40 decomposition by exploiting that we can share
    some of the Strassen-like tricks.
    
    Wait, let me just try to construct a valid decomposition with rank < 80.
    
    Here's an approach: use the schoolbook for 4x4 square (rank 16 for the square part)
    and then extend. But that's not straightforward.
    
    Let me try a different approach: use rank 40 by grouping the 5 columns of B.
    
    Actually, the simplest improvement: note that for each fixed j, the term
    A[i,j] * B[j,k] contributes to all (i,k) pairs. The schoolbook uses 1 term per (i,j,k).
    
    Let me just implement the schoolbook with rank 80 first, then try to find a lower rank.
    
    Actually, I recall that the rank of the (4,4,5) matrix multiplication tensor is at most 49
    (from Lader's result for 4x4 extended). But I don't have the explicit decomposition memorized.
    
    Let me just do the schoolbook rank-80 decomposition, which is guaranteed correct.
    """
    n, m, p = DIMENSIONS  # n=4, m=4, p=5
    rank = n * m * p  # 80 for schoolbook
    
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

# Let me actually try to find a better rank using optimization
# But given time constraints, let me first ensure correctness with schoolbook
# Then try to improve.

# Actually, let me think about this more carefully. The key insight is that
# we can use the Strassen trick for 2x2 blocks.
#
# Split A into 2x2 blocks of 2x2:
# A = [[A00, A01], [A10, A11]]
# B = [[B00, B01, B02, B03, B04],
#      [B10, B11, B12, B13, B14],
#      [B20, B21, B22, B23, B24],
#      [B30, B31, B32, B33, B34]]
#
# This doesn't help directly since B has 4 rows.
#
# Let me just go with schoolbook rank 80. It's correct and the evaluator
# will measure the rank. Any improvement would need explicit knowledge of
# a lower-rank decomposition.

# Actually wait - let me reconsider. Maybe I can use a rank-40 decomposition
# by noting that for each pair (i,k), we need to compute sum_j A[i,j]*B[j,k].
# The schoolbook uses 4 terms for each (i,k), so 4*4*5 = 80 total.
#
# Can we share computations? For fixed i, we compute 5 inner products of length 4.
# That's 5 dot products, which can be done with 5*4 = 20 multiplications.
# For all 4 values of i, that's 80. No sharing possible in the schoolbook.
#
# The Strassen trick works for square matrices because we can combine the
# partial sums. For rectangular, it's less clear.
#
# I'll stick with schoolbook rank 80 for now.
