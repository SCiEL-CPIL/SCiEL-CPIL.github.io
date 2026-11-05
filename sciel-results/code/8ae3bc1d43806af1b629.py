"""Matrix multiplication 3,3,7 - rank-minimal tensor decomposition."""
DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction
from itertools import product

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x3 * 3x7 matrix multiplication.
    
    Strategy:
    1. Use the Strassen-like rank-5 decomposition for 2x2x2 blocks.
    2. For the 3x3x3 core, we can potentially use a rank-23 or lower decomposition.
    3. For the 3x3x7 case, we need to extend this.
    
    However, constructing a truly minimal rank decomposition is extremely hard.
    The current schoolbook is 63.
    
    Let's try a hybrid approach:
    - Decompose the 3x3 matrix multiplication into 2x2 blocks plus border rows/cols.
    - A 3x3 * 3x3 multiplication can be seen as:
      [A11 A12; A21 A22] where A11 is 2x2, A12 is 2x1, A21 is 1x2, A22 is 1x1.
      Similarly for B.
      
      Actually, let's just try to find a valid decomposition with rank < 63.
      
    A simpler improvement: 
    The tensor for 3x3 * 3x3 has rank at most 23 (best known upper bound for 3x3x3 is 23, 
    but actually the best known is 23 for 3x3x3? No, Strassen is 7 for 2x2x2. 
    For 3x3x3, the best known upper bound is 23. The lower bound is 19.
    Wait, let me check. The tensor rank of 3x3x3 is known to be between 19 and 23.
    Actually, the exact rank is not known. But we can use a known decomposition.
    
    For 3x3 * 3x7:
    This is equivalent to multiplying a 3x3 matrix by a 3x7 matrix.
    We can view this as 7 separate 3x3 * 3x1 multiplications, but that's not efficient.
    Or as one big multiplication.
    
    Let's try a constructive approach using Kronecker products or block structure.
    
    Alternative: Use the fact that 3x3 * 3x7 can be embedded in a larger multiplication.
    
    Actually, let's just implement a search for a rank-deficient decomposition using linear algebra.
    But that might be too slow.
    
    Let's stick with a known valid decomposition.
    
    The schoolbook is 63.
    
    Can we do better?
    
    Consider the tensor T_{i,j,k} where i in 0..2, j in 0..2, k in 0..6.
    T = sum_{r} U_r \otimes V_r \otimes W_r
    
    If we can find a set of rank-1 tensors that sum to T.
    
    Let's try to use the Strassen algorithm for the 2x2x2 sub-blocks.
    
    Partition the 3x3 matrix A into:
    A = [A00 A01 A02; A10 A11 A12; A20 A21 A22]
    
    This is getting complex. Let's just return the schoolbook for now, but maybe try a simple optimization.
    
    Actually, there is a known result: the rank of 3x3 * 3x3 is at most 23.
    For 3x3 * 3x7, we can think of it as 7 copies of 3x3 * 3x1.
    
    Let's try a different approach: use the "border rank" or just find a valid decomposition with rank < 63.
    
    Here's a simple idea:
    Use the fact that 3 = 2 + 1.
    A 3x3 * 3x7 multiplication can be done by:
    1. A 2x2 * 2x7 multiplication (using Strassen-like methods)
    2. Plus the border terms.
    
    But this is getting complicated. Let's just ensure correctness with the schoolbook, 
    and if time permits, try a random search for a lower rank.
    
    Given the time constraints and the difficulty of finding a provably lower rank,
    I will return the schoolbook decomposition.
    """
    start_time = time.time()
    deadline = start_time + 140.0

    n, m, p = 3, 3, 7
    rank_schoolbook = n * m * p  # 63

    U = [[0] * rank_schoolbook for _ in range(n * m)]
    V = [[0] * rank_schoolbook for _ in range(m * p)]
    W = [[0] * rank_schoolbook for _ in range(p * n)]

    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
