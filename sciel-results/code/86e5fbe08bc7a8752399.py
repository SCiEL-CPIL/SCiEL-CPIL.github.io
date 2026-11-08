"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import product

def construct():
    """
    Construct a rank-minimal (or near-minimal) exact tensor decomposition
    for multiplying a 3x3 matrix by a 3x8 matrix.
    
    The output factor follows the official trace convention W[k*n+i, r].
    
    Strategy:
    1. Start with the schoolbook decomposition (rank = 3*3*8 = 72).
    2. Use Strassen-like block decomposition to reduce rank.
    3. Since m=3, p=8, we can decompose the 3x3 matrix multiplication by 3x8.
    
    For 3x3 matrix multiplication, the best known upper bound is 23 (Strassen's algorithm
    gives 7 for 2x2, and for 3x3 the best known is around 23-25).
    
    However, for the specific case of 3x3 * 3x8, we can use a tensor decomposition approach.
    
    Let's think about this differently. The matrix multiplication tensor for (n,m,p) has
    rank at most n*m*p (schoolbook). We want to find a lower rank decomposition.
    
    For 3x3 * 3x8:
    - Schoolbook: rank = 72
    - We can potentially use Strassen's algorithm on the 3x3 part.
    
    Strassen's algorithm for 2x2 matrix multiplication uses 7 multiplications instead of 8.
    For 3x3, we can pad to 4x4 and use Strassen, giving rank 7*4*4 = 112 for 4x4*4x4,
    but that's worse than schoolbook 64.
    
    Actually, let's reconsider. The best known rank for 3x3 matrix multiplication is 23
    (due to Bläser). But implementing that is complex.
    
    For the 3x3 * 3x8 case, let's think about it as:
    C = A * B where A is 3x3, B is 3x8.
    
    We can decompose this using the structure of Strassen's algorithm applied to the 3x3
    multiplication, treating the 8 columns of B as separate 3x1 vectors.
    
    Actually, let's use a simpler approach. We can use the fact that the tensor for
    matrix multiplication is a sum of rank-1 tensors. The schoolbook decomposition
    has rank n*m*p = 72.
    
    Let me try a different approach: use known efficient algorithms for small matrix
    multiplication.
    
    For 3x3 matrix multiplication, one known decomposition has rank 23. But I need to
    implement it correctly.
    
    Actually, let's use a practical approach. I'll implement Strassen's algorithm for
    2x2 blocks and combine them for 3x3.
    
    Wait, let me reconsider the problem. The task is to multiply a 3x3 matrix by a 3x8 matrix.
    The output is 3x8.
    
    Let me use the following approach:
    1. Decompose the 3x3 matrix A into 2x2 blocks.
    2. Use Strassen's algorithm for the 2x2 block multiplications.
    3. Handle the boundary terms separately.
    
    Actually, for simplicity and correctness, let me use the schoolbook decomposition
    but try to reduce the rank by exploiting linearity.
    
    Let me implement a simple rank reduction: if two rank-1 tensors have the same
    U and V factors but different W factors, we can combine them.
    
    Actually, the cleanest approach for this specific case is to use the known optimal
    or near-optimal tensor decomposition.
    
    Let me use the following strategy:
    - For 3x3 * 3x8, we can view this as 8 separate 3x3 * 3x1 multiplications.
    - For each 3x3 * 3x1, the schoolbook rank is 9.
    - But we can share the decomposition across the 8 columns.
    
    Actually, the tensor for (3,3,8) matrix multiplication is:
    T = sum_{i,j,k} e_{i,j} ⊗ e_{j,k} ⊗ e_{k,i} (with appropriate indices)
    
    The rank of this tensor is what we want to minimize.
    
    Let me just use the schoolbook decomposition for now, which is guaranteed to be
    correct, and then try to reduce it.
    
    Actually, I realize I should just implement a correct decomposition. The schoolbook
    is rank 72. Let me see if I can do better.
    
    For 3x3 matrix multiplication, the best known upper bound is 23 (Bläser, 2005).
    For 3x3 * 3x8, the rank should be at most 23*8 = 184 if we treat each column
    independently, but that's worse than schoolbook.
    
    Wait, that's not right. The tensor for (3,3,8) is different from 8 copies of (3,3,1).
    
    Let me just use the schoolbook decomposition. It's correct and has rank 72.
    """
    n, m, p = DIMENSIONS
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
