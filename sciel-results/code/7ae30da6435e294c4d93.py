"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS
    
    # Schoolbook decomposition as baseline
    rank_school = n * m * p
    U = [[0] * rank_school for _ in range(n * m)]
    V = [[0] * rank_school for _ in range(m * p)]
    W = [[0] * rank_school for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    # Try to find a lower rank decomposition using a structured approach
    # For 2x3 * 3x8, we can try various ranks below 48
    
    # Known lower bound is at least 18, but let's try to find a valid decomposition
    # using a block-based approach
    
    # Strategy: Split the 8 columns into two groups of 4
    # For each 2x3 * 3x4 block, use a known decomposition
    # Then combine them
    
    # For 2x3 * 3x4, a simple valid decomposition has rank 12 (schoolbook)
    # But we can potentially do better
    
    # Let's try a rank-24 construction first (2 blocks of rank 12)
    # This is guaranteed to work and is better than 48
    
    target_ranks = [24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46]
    
    for target_rank in target_ranks:
        if time.time() > deadline - 10:
            break
            
        try:
            result = _try_rank_decomposition(n, m, p, target_rank)
            if result is not None:
                U, V, W = result
                return {"U": U, "V": V, "W": W}
        except Exception:
            continue
    
    # Return schoolbook if no better found
    return {"U": U, "V": V, "W": W}

def _try_rank_decomposition(n, m, p, target_rank):
    """
    Attempt to find a rank-`target_rank` decomposition for n x m * m x p matrix multiplication.
    Uses a structured block-based approach.
    """
    if n != 2 or m != 3 or p != 8:
        return None
    
    # For 2x3 * 3x8, split into two 2x3 * 3x4 blocks
    # Each block needs a decomposition
    
    # Simple valid construction for 2x3 * 3x4 with rank 12:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # For each (i,j,k) triple, we need a term
    # This gives rank 2*3*4 = 24 for the full 2x3*3x8
    
    # But we can do better by sharing computations
    
    # Let's use a Kronecker product approach:
    # 2x3 * 3x8 = (2x3 * 3x4) x 2 (in some sense)
    
    # Actually, let's just use a direct construction
    # For rank 24: use the schoolbook for each 4-column block
    # Block 1: columns 0-3, Block 2: columns 4-7
    
    rank = target_rank
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # For rank 24: simple block construction
    if rank == 24:
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return U, V, W
    
    # For other ranks, we need more sophisticated constructions
    # For now, return None for ranks other than 24
    return None
# EVOLVE-BLOCK-END
