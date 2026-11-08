"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 5]
import time
import numpy as np
from itertools import product

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 120
    
    # Schoolbook as fallback
    rank_schoolbook = n * m * p
    U_school = [[0] * rank_schoolbook for _ in range(n * m)]
    V_school = [[0] * rank_schoolbook for _ in range(m * p)]
    W_school = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    # Try to find a better decomposition using Winograd-style optimizations
    # For 4x4 * 4x5, we can use block structure to reduce rank
    
    best_result = None
    best_rank = rank_schoolbook
    
    # Try various structured decompositions
    # Strategy: use block-Winograd to reduce rank
    
    # For 4x4 * 4x5, we can split into 2x2 blocks
    # Each 2x2 * 2x2 block has rank 7 (Winograd) instead of 8
    # We have a 4x4 result, so 4 blocks of 2x2, plus handling the 5th column
    
    # Let's try a concrete construction:
    # Split A (4x4) into 2x2 blocks, B (4x5) into 2x2 and 2x3 blocks
    
    # Actually, let's use a more systematic approach:
    # We'll try to find a rank-r decomposition by solving the tensor equation
    
    # For rank r, we need matrices U (16 x r), V (20 x r), W (20 x r)
    # such that for all i,j,k: sum_t U[i*4+j,t] * V[j*5+k,t] * W[k*4+i,t] = 1
    
    # Let's try ranks from 72 down to 60 with a greedy approach
    
    for target_rank in range(72, 59, -1):
        if time.time() > deadline - 5:
            break
        
        found = _find_decomposition_structured(n, m, p, target_rank, deadline - 5)
        if found is not None:
            U, V, W = found
            best_rank = target_rank
            best_result = {"U": U, "V": V, "W": W}
            break
    
    if best_result is not None:
        return best_result
    else:
        return {"U": U_school, "V": V_school, "W": W_school}

def _find_decomposition_structured(n, m, p, rank, time_limit):
    """Find a rank decomposition using structured Winograd approach."""
    deadline = time.time() + time_limit
    
    # For 4x4 * 4x5, use block-Winograd
    # Split into 2x2 blocks for the 4x4 part
    
    # Winograd for 2x2 * 2x2: rank 7
    # Transform matrices for 2x2:
    # A' = [[a00+a01, a00-a01, a10+a11, a10-a11, a00, a11, a01], ...]
    # Actually, standard Winograd 2x2:
    # m0 = a00+a01, m1 = a00-a01, m2 = a10+a11, m3 = a10-a11
    # m4 = a00, m5 = a11, m6 = a01
    # Then multiply with B transformed similarly
    
    # For our case, let's construct explicitly
    # We'll use a combination of schoolbook and Winograd reductions
    
    # Initialize with zeros
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Use Winograd for 2x2 blocks where possible
    # For the 4x4 * 4x4 part (first 4 columns of B), use 4 blocks of 2x2*2x2
    # Each block uses rank 7, so 4*7 = 28 for the 4x4*4x4 part
    
    # For the 4x4 * 4x1 part (5th column), use schoolbook: 16
    
    # Total: 28 + 16 = 44? That seems too low. Let me reconsider.
    
    # Actually, Winograd 2x2*2x2 has rank 7, but we need to handle the full 4x4*4x5
    
    # Let me use a different approach: direct construction with known patterns
    
    # For simplicity, let's try to find a valid decomposition by solving the system
    # using a greedy assignment
    
    # The constraint is: for all (i,j,k), sum_t U[i*m+j,t]*V[j*p+k,t]*W[k*n+i,t] = 1
    
    # We'll try to satisfy these constraints one by one
    
    # First, let's try a sparse construction where each (i,j,k) is satisfied by exactly one term
    
    # This is essentially finding a perfect matching in a 3D hypergraph
    
    # For now, let's return None and use schoolbook
    return None

# EVOLVE-BLOCK-END
