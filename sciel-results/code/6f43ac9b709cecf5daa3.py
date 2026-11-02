"""Matrix multiplication 4x5x5 rank-minimal decomposition."""
import numpy as np
from fractions import Fraction
import time
import sys

DIMENSIONS = [4, 5, 5]

def construct():
    """
    Construct a tensor decomposition for multiplying a 4x5 matrix by a 5x5 matrix.
    
    We use a structural approach: exploit the rectangular nature (4x5 * 5x5) by
    treating the 4 rows as two groups of 2, and applying Strassen-like
    decompositions on 2x2 blocks where possible, combined with careful
    handling of the 5-column/5-row structure.
    
    The key insight: for 4x5 * 5x5, we can view this as:
    - Split A (4x5) into two 2x5 blocks: A_top, A_bottom
    - B (5x5) stays whole
    - C (4x5) = [C_top; C_bottom] where C_top = A_top @ B (2x5), C_bottom = A_bottom @ B (2x5)
    
    Each 2x5 * 5x5 multiplication can potentially be done with fewer than 2*5*5=50
    multiplications using structured decompositions.
    
    However, a more promising approach: use the fact that the matmul tensor for
    n x m * m x p has known lower bounds and upper bounds. For 4x5x5:
    - Schoolbook: rank 100
    - We can try to find a decomposition with rank < 100
    
    Strategy: Use a hybrid approach where we:
    1. Start with schoolbook (rank 100)
    2. Try to merge terms using linear algebra over the tensor space
    3. Use the specific structure of 4x5x5 to find redundancies
    
    Actually, let me try a concrete construction. For 2x2x2, Strassen gives rank 7
    instead of 8. The savings come from exploiting the algebraic structure.
    
    For 4x5x5, one approach is to use "blocking":
    - View 4x5 as 2 blocks of 2x5
    - View 5x5 as having structure
    
    Let me try a different concrete approach: use the fact that for certain
    rectangular matrices, we can use "peeling" or "unfolding" techniques.
    
    Specifically, for 4x5 * 5x5, consider:
    C[i,k] = sum_j A[i,j] B[j,k] for i in 0..3, k in 0..4
    
    This is 20 output entries, each requiring a sum of 5 products.
    
    One known technique: if we can find a set of linear forms in A and B whose
    products generate all the needed C entries, we can reduce the rank.
    
    Let me try to implement a search-based approach that attempts to find
    a lower-rank decomposition by:
    1. Starting with schoolbook
    2. Trying to express one term as a combination of others (tensor rank reduction)
    
    But this is computationally expensive. Let me instead try a known
    constructive approach.
    
    For 4x4x4, there are known decompositions with rank < 64. For 4x5x5,
    let me try to use a block decomposition.
    
    Alternative: use the "lattice method" or "embedding" approach.
    
    Actually, let me just try to implement a simple optimization:
    - Represent the tensor as a 3D array T[i,j,k] = 1 for all valid (i,j,k)
    - Wait, no. The matmul tensor is T[i,j,k] = 1 if and only if the triple
      (i,j,k) is "valid" for matmul, which is always true. But the tensor
      lives in (R^{20})^* ⊗ (R^{25})^* ⊗ (R^{20})^* with specific basis.
    
    Let me just go with a practical approach: try rank 99, 98, ... by
    attempting to find if any schoolbook term is redundant.
    
    For now, return schoolbook (rank 100) as the safe baseline.
    """
    n, m, p = DIMENSIONS
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110  # 110s out of 150s budget
    
    # Schoolbook baseline
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
    
    # Try to improve using a structured approach
    # For 4x5x5, try to use block structure to reduce rank
    
    # Approach: treat the problem as computing C = A @ B where A is 4x5, B is 5x5
    # Split into 2x2 blocks of rows in A: A = [A1; A2] where A1, A2 are 2x5
    # Then C = [C1; C2] where C1 = A1 @ B, C2 = A2 @ B
    # Each 2x5 * 5x5 can potentially be done with < 50 mults
    
    # For 2x5 * 5x5, schoolbook is 50. Can we do better?
    # 2x5 * 5x5: C[i,k] = sum_j A[i,j] B[j,k] for i in 0,1; k in 0..4
    # That's 10 outputs, each a sum of 5 products.
    
    # One idea: use the fact that B is 5x5 and try to find linear combinations
    # of rows of B that can be reused.
    
    # Let me try a concrete small-rank construction for 2x5*5x5:
    # If we could do it in rank 45, then 4x5*5x5 would be rank 90.
    
    # For now, just return schoolbook.
    return {"U": U, "V": V, "W": W}
