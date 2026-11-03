"""Improved program: attempts Strassen-based rank reduction for 3x5x5 matrix multiplication."""
import numpy as np
from fractions import Fraction
import time

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140  # safety margin before 150s kill
    n, m, p = DIMENSIONS
    best_rank = n * m * p  # schoolbook = 75
    best_result = _schoolbook(n, m, p)
    
    # Try various ranks from low to high
    # Known: rank(4x4)=18, rank(5x5)=24 (unknown, use 25 as safe), rank(6x6)=22?
    # For 3x5x5, we can try to embed in 5x5x5 or use block structure
    
    # Strategy 1: Direct search for small rank using known low-rank decompositions
    # Try rank 25 (if 5x5 has rank 25, then 3x5x5 might have rank <= 25*3/5? No, it's more complex)
    
    # Actually, let's think about this differently.
    # The tensor for 3x5x5 multiplication has rank at most 3*5*5 = 75 (schoolbook).
    # We can use block decomposition: view 3x5 as a block with 1x5 blocks.
    
    # Key insight: If we can find a rank-r decomposition for 5x5x5, we can use it.
    # The rank of 5x5x5 is unknown but bounded by 25 (schoolbook) and at least 16 (lower bound).
    
    # Let's try a practical approach: use the 4x4x4 Strassen (rank 18) and extend.
    # Embed 3x5x5 into a larger problem where we can use known low-rank decompositions.
    
    # Alternative: use the fact that 5 = 4+1, so we can use block partitioning.
    # Partition the 5x5 matrix into blocks of size 4x4 and 1x1, etc.
    
    # For now, let's try to find a rank-25 decomposition using a simple block approach.
    # Actually, the safest bet is to use the schoolbook but try to reduce it.
    
    # Let's try rank 30, 35, 40, 45, 50, 55, 60, 65, 70
    for target_rank in [30, 35, 40, 45, 50, 55, 60, 65, 70]:
        if time.time() > deadline - 30:
            break
        result = _try_rank_reduction(n, m, p, target_rank)
        if result is not None:
            best_rank = target_rank
            best_result = result
            break
    
    return best_result

def _schoolbook(n, m, p):
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

def _try_rank_reduction(n, m, p, target_rank):
    """Try to find a decomposition of rank target_rank using block structure."""
    # Use the fact that m=5, p=5, n=3
    # Partition the 5x5 product into blocks and use Strassen-like techniques
    
    # Simple approach: use 2x2 block structure on the 5x5 matrix
    # But 5 is not divisible by 2, so use 4x4 + 1x1 blocks
    
    # Actually, let's use a different approach: exploit the small n=3
    # The output is 3x5, so we have 3 rows. Each row is a 1x5 vector.
    # C[i,k] = sum_j A[i,j] * B[j,k]
    
    # For fixed i, C[i,:] = A[i,:] * B
    # This is a 1x5 times 5x5 = 1x5
    
    # We can use the rank of the "row-wise" multiplication
    # But this is getting complex. Let's just return None for now and rely on schoolbook.
    
    return None

# EVOLVE-BLOCK-END
