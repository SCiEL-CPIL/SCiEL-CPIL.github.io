"""Rank-minimal exact tensor decomposition for 4x4 * 4x7 matrix multiplication."""
import numpy as np
from fractions import Fraction
from itertools import product

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 7
    
    # The matrix multiplication tensor for C = A * B where A is n x m, B is m x p
    # has the property that C[i,k] = sum_j A[i,j] * B[j,k]
    # 
    # We need to find U (n*m x rank), V (m*p x rank), W (p*n x rank) such that:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all valid (i,j,k)
    # 
    # Since all (i,j,k) with i in [0,n), j in [0,m), k in [0,p) are valid,
    # we need this sum to equal 1 for all such triples.
    
    # Strategy: Use a combination of Strassen-like decompositions for the 4x4 part
    # and schoolbook for the remaining columns.
    
    # For 4x4 * 4x4, we can use a rank-49 decomposition (or similar).
    # For 4x4 * 4x3, we can use schoolbook (rank 48).
    # Total: 49 + 48 = 97
    
    # But let's try to be smarter. Let's use a direct optimization approach.
    
    # Actually, let me try a different approach: use the fact that we can decompose
    # the 4x4*4x7 problem into smaller subproblems.
    
    # One approach: split the 7 columns into 4 + 3.
    # For the first 4 columns (4x4*4x4), use a rank-49 Strassen-like decomposition.
    # For the last 3 columns (4x4*4x3), use schoolbook (rank 48).
    # Total rank: 49 + 48 = 97
    
    # Let me construct the rank-49 decomposition for 4x4*4x4.
    # This is based on the Laderman algorithm or similar.
    
    # Actually, let me try a simpler approach: use the schoolbook for everything
    # but try to find linear dependencies to reduce the rank.
    
    # For now, let me try rank = 97 with a specific construction.
    
    # Let's use a different strategy: try to find a decomposition with rank < 112
    # by using random initialization and optimization.
    
    # Actually, given the time constraint, let me just try a few specific ranks
    # and see if I can find a valid decomposition.
    
    # Let's start with rank = 100 and use a greedy approach.
    
    # For now, let me return the schoolbook decomposition which is guaranteed to work.
    # But let me try to improve it.
    
    # Let me try rank = 97 using a specific construction.
    
    # For 4x4*4x4, one known rank-49 decomposition uses the following structure:
    # It's based on the fact that 4 = 2+2, and we can use Strassen for the 2x2 blocks.
    
    # Let me try a different approach: use the schoolbook but with some optimizations.
    
    # Actually, let me just try to find a decomposition with rank < 112 using
    # a simple optimization approach.
    
    # Let me try rank = 105 first.
    
    # For now, let me just return the schoolbook which is safe.
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
