"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = DIMENSIONS  # 3, 4, 8
    nm, mp, pn = n * m, m * p, p * n  # 12, 32, 24
    
    # Construct the matrix multiplication tensor T of shape (nm, mp, pn)
    # T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((nm, mp, pn), dtype=np.int8)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1
    
    # Schoolbook baseline
    rank_school = n * m * p  # 96
    U_school = np.zeros((nm, rank_school), dtype=np.int8)
    V_school = np.zeros((mp, rank_school), dtype=np.int8)
    W_school = np.zeros((pn, rank_school), dtype=np.int8)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1
                V_school[j*p+k, col] = 1
                W_school[k*n+i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = rank_school
    
    def validate(U, V, W):
        """Check if U, V, W correctly decompose T."""
        rank = U.shape[1]
        T_recon = np.zeros((nm, mp, pn), dtype=np.int64)
        for r in range(rank):
            T_recon += np.einsum('i,j,k->ijk', U[:,r].astype(np.int64), 
                                  V[:,r].astype(np.int64), W[:,r].astype(np.int64))
        return np.array_equal(T_recon, T.astype(np.int64))
    
    # Try to find a lower rank decomposition using structured approaches
    # Strategy: Use block decomposition and Strassen-like ideas
    
    # Approach 1: Try to use the fact that 8 = 2*4
    # Split p into two groups of 4
    # This gives R(3,4,8) <= 2 * R(3,4,4)
    # For R(3,4,4), schoolbook is 48, so this gives 96 (same as schoolbook)
    # But maybe R(3,4,4) < 48?
    
    # Approach 2: Try random search for lower rank
    # This is expensive but might find something
    
    # For now, let's try a few structured decompositions
    
    # Try rank 90: see if we can merge some terms
    # The idea is that some combinations of rank-1 terms might cancel out
    
    # Actually, let's try a more systematic approach:
    # Use the known fact that for small matrices, sometimes we can do better
    
    # For 2x2x2, Strassen gives rank 7 vs schoolbook 8
    # For 3x3x3, best known is 23 vs schoolbook 27
    
    # For 3x4x8, let's try to see if we can get below 96
    
    # Try a greedy approach: start with schoolbook and try to reduce
    # This is complex, so let's just return schoolbook for now
    
    # But let's try one more thing: see if there's a simple pattern
    # that allows us to use fewer terms
    
    # For now, return the schoolbook decomposition
    return {
        "U": U_school.tolist(),
        "V": V_school.tolist(),
        "W": W_school.tolist()
    }
# EVOLVE-BLOCK-END
