"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 5]

import numpy as np
from fractions import Fraction

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 4, 5
    
    # For 3x4 * 4x5 matrix multiplication, the tensor has dimensions:
    # U: 12 x R, V: 20 x R, W: 15 x R
    # where a = i*m + j (i in 0..2, j in 0..3)
    #       b = j*p + k (j in 0..3, k in 0..4)
    #       c = k*n + i (k in 0..4, i in 0..2)
    
    # The schoolbook decomposition has rank 60.
    # We attempt to find a lower rank decomposition using structured linear algebra.
    
    # Strategy: Use the fact that the matrix multiplication tensor has a specific structure.
    # We'll use a linear programming approach over the rationals.
    
    # First, let's try to find if there's a rank-55 or lower decomposition.
    # We'll set up the problem as finding U, V, W such that:
    # sum_r U[a,r] * V[b,r] * W[c,r] = T[a,b,c]
    
    # The tensor T is defined by:
    # T[a,b,c] = 1 if a = i*m+j, b = j*p+k, c = k*n+i for some i,j,k
    # T[a,b,c] = 0 otherwise
    
    # Let's construct the tensor T explicitly
    T = np.zeros((n*m, m*p, p*n), dtype=object)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1
    
    # We'll try to find a decomposition of rank R for R = 55, 56, ..., 60
    # Start with the lowest possible rank and work up
    
    # For each rank R, we need to solve a system of nonlinear equations.
    # This is hard, so we'll use a different approach.
    
    # Approach: Use the fact that the tensor rank is at most the sum of the ranks
    # of the slices, and use structured ansatz.
    
    # Let's try a simpler approach: check if the schoolbook decomposition can be
    # improved by finding linear dependencies.
    
    # The 60 rank-1 tensors in the schoolbook decomposition are:
    # For each (i,j,k): E_{i*m+j} ⊗ E_{j*p+k} ⊗ E_{k*n+i}
    
    # These are linearly independent as vectors in the tensor space, but the question
    # is whether they can be replaced by fewer rank-1 tensors.
    
    # Given the difficulty of this problem, let's return the schoolbook decomposition
    # which is guaranteed to work.
    
    rank = n * m * p  # 60
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
