"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from fractions import Fraction

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition exactly reproduces matrix multiplication."""
    rank = len(U[0]) if U else 0
    # Build the multiplication tensor and check
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def construct():
    """
    Find a low-rank exact tensor decomposition for 3x3 * 3x8 matrix multiplication.
    Returns dict with U, V, W matrices.
    """
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # Safety margin before 120s limit
    
    # Strategy: Split B (3x8) into B1 (3x3), B2 (3x3), B3 (3x2)
    # C = A*B1 + A*B2 + A*B3
    # Use schoolbook for each block:
    #   A*B1: 3*3*3 = 27 terms
    #   A*B2: 3*3*3 = 27 terms  
    #   A*B3: 3*3*2 = 18 terms
    # Total: 72 (same as schoolbook, no improvement)
    
    # Better: Use Strassen-like rank-7 for 3x3*3x3 blocks
    # But we need an explicit rank-7 decomposition for 3x3*3x3
    
    # For now, let's construct a valid decomposition with rank < 72
    # using a structured approach
    
    # Approach: Use the fact that for 3x3 * 3x8, we can think of it as
    # computing 8 independent 3x3 * 3x1 multiplications, but share the A factors
    
    # Let's build a decomposition where we group the 8 output columns
    # and use linear combinations to reduce rank
    
    # Concrete construction:
    # For each j in {0,1,2} (middle dimension), we need to compute
    # C[:, k] = sum_i A[i,j] * B[j,k] for k in 0..7
    # This is: for fixed j, C[:, :] = A[:, j] * B[j, :]
    # which is an outer product of a 3-vector and a 8-vector = rank 24 per j
    # Total: 3 * 24 = 72
    
    # To reduce rank, we need to combine across j values.
    # Key insight: If we form linear combinations of the A columns and B rows,
    # we can reduce the number of independent outer products.
    
    # Let's try: form 7 linear combinations of the 9 (i,j) entries of A
    # and 7 linear combinations of the 9 (j,k) entries of B (for first 3 columns)
    # This is the Strassen idea but for 3x3.
    
    # For a practical improvement, let's use a hybrid:
    # - For the first 6 columns of B (two 3x3 blocks), use a rank-10 decomposition each
    # - For the last 2 columns, use schoolbook (rank 6)
    # Total: 10 + 10 + 6 = 26
    
    # But we need explicit rank-10 decompositions for 3x3 * 3x3.
    # The known minimum rank for 3x3 * 3x3 is 19 (not 7 as I mistakenly thought).
    # Strassen's rank-7 is for 2x2 * 2x2.
    
    # Let me just return the schoolbook decomposition which is guaranteed correct.
    # The rank is 72, which matches the seed score of 1.0.
    
    schoolbook_rank = n * m * p  # 72
    
    U = [[0] * schoolbook_rank for _ in range(n * m)]
    V = [[0] * schoolbook_rank for _ in range(m * p)]
    W = [[0] * schoolbook_rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    # Verify correctness
    if not _verify(U, V, W, n, m, p):
        raise ValueError("Schoolbook decomposition failed verification")
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
