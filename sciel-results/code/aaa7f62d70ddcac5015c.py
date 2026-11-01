"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]
import time
import random
import numpy as np
from itertools import product

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2x4 matrix by a 4x6 matrix.
    The tensor T has indices (i, j, k) with i in [0, n), j in [0, m), k in [0, p).
    T[i,j,k] = 1 for all valid i,j,k.
    The decomposition is T = sum_r U_r * V_r * W_r where:
    U is (n*m, rank), V is (m*p, rank), W is (p*n, rank).
    The entry T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1.
    
    We want to minimize rank. We'll use a systematic search with time budget.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145  # Leave 5s safety margin
    
    # Generate schoolbook solution as baseline
    def schoolbook():
        r = n * m * p
        U = [[0] * r for _ in range(n * m)]
        V = [[0] * r for _ in range(m * p)]
        W = [[0] * r for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return U, V, W, r
    
    U_sb, V_sb, W_sb, r_sb = schoolbook()
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = r_sb
    
    # Try to find lower rank solutions using randomized search
    # The constraint is: for all (i,j,k), sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # We'll try ranks from 47 down to some minimum
    # For each rank, we'll do random initialization and local search
    
    def verify_decomposition(U, V, W, r):
        """Check if the decomposition is exact."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0
                    for rr in range(r):
                        val += U[i*m+j][rr] * V[j*p+k][rr] * W[k*n+i][rr]
                    if abs(val - 1) > 1e-9:
                        return False
        return True
    
    # Try random search for lower rank
    for target_rank in range(47, 30, -1):
        if time.time() > deadline:
            break
        
        # Try multiple random initializations
        for attempt in range(50):
            if time.time() > deadline:
                break
            
            # Random initialization with small values
            U = [[random.choice([0, 1, -1, 0.5, -0.5]) for _ in range(target_rank)] for _ in range(n*m)]
            V = [[random.choice([0, 1, -1, 0.5, -0.5]) for _ in range(target_rank)] for _ in range(m*p)]
            W = [[random.choice([0, 1, -1, 0.5, -0.5]) for _ in range(target_rank)] for _ in range(p*n)]
            
            if verify_decomposition(U, V, W, target_rank):
                best_result = {"U": U, "V": V, "W": W}
                best_rank = target_rank
                break
        
        if best_rank < target_rank:
            break
    
    return best_result
# EVOLVE-BLOCK-END
