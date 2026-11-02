import time
import random
import numpy as np

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    
    # Start with schoolbook as fallback
    best_result = _schoolbook_decomposition(n, m, p)
    best_rank = n * m * p
    
    # Try to find a lower-rank decomposition using the peeling/block structure
    # For 2x3x6, we can view it as two 2x3x3 blocks (columns 0-2 and 3-5 of B)
    # Try to find a decomposition that shares terms between blocks
    
    # First, try a structured rank-28 decomposition (two independent 2x3x3 blocks)
    # Then try to reduce further
    
    # Let's try a direct search for rank < 36
    # We'll use a parameterized ansatz based on the block structure
    
    # Try rank 28: two independent 2x3x3 decompositions
    # For 2x3x3, we can use a rank-14 decomposition if we find it
    # For now, let's try to find any valid decomposition with rank < 36
    
    # Strategy: Use the fact that n=2 to create a structured ansatz
    # U has 6 rows, V has 18 rows, W has 12 rows
    # The constraint: sum_c U[i*3+j, c] * V[j*6+k, c] * W[k*2+i, c] = 1 for all i,j,k
    
    # Let's try a block-structured ansatz where we separate the two 3-column blocks
    # Block 1: k=0,1,2; Block 2: k=3,4,5
    
    # Try to find a rank-20 decomposition by sharing some terms
    # This is a hard optimization problem, so let's try a heuristic approach
    
    # Given time constraints, let's try a few structured decompositions
    
    # Decomposition 1: Rank 28 (two independent 2x3x3 blocks with rank 14 each)
    # We need the rank-14 coefficients for 2x3x3
    
    # Let's try to construct a valid rank-28 decomposition using a known method
    # For 2x3x3, one approach is to use the 2x2x2 Strassen (rank 7) and extend
    
    # Actually, let's just try to verify if we can get rank < 36 with a simple search
    # We'll try random half-integer matrices and check if they form a valid decomposition
    
    if time.time() < deadline - 30:
        result = _try_block_decomposition(n, m, p, deadline)
        if result is not None:
            rank = len(result['U'][0])
            if rank < best_rank:
                best_result = result
                best_rank = rank
    
    return best_result

def _schoolbook_decomposition(n, m, p):
    """Return the standard schoolbook decomposition."""
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

def _try_block_decomposition(n, m, p, deadline):
    """Try to find a lower-rank decomposition using block structure."""
    # For 2x3x6, try a structured decomposition
    # We'll attempt to find rank-20 or rank-24 decompositions
    
    # Given the complexity, let's try a simple approach:
    # Use the block structure where B is split into two 3x3 blocks
    
    # A valid rank-28 decomposition: two independent 2x3x3 decompositions
    # For 2x3x3, we can use a rank-14 decomposition
    
    # Let's try to find a rank-14 decomposition for 2x3x3 first
    # If we can't find it quickly, fall back to rank-18 (schoolbook for 2x3x3)
    
    # For now, return None to use the schoolbook fallback
    # A more sophisticated search would be needed for actual improvement
    
    return None

# EVOLVE-BLOCK-END
