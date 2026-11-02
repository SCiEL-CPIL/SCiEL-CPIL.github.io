"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 6]

import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3x5 matrix by a 5x6 matrix.
    
    The output factor follows the official trace convention W[k*n+i, r].
    Returns {'U': (n*m, rank), 'V': (m*p, rank), 'W': (p*n, rank)} with integer/half-integer entries.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Schoolbook decomposition as fallback
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    best_result = schoolbook()
    best_rank = n * m * p
    
    # Verify decomposition: T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] should equal 1
    def verify_decomposition(U, V, W, rank):
        """Verify that the decomposition exactly matches matrix multiplication."""
        if len(U) != n * m or len(U[0]) != rank:
            return False
        if len(V) != m * p or len(V[0]) != rank:
            return False
        if len(W) != p * n or len(W[0]) != rank:
            return False
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(rank):
                        s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                    if abs(s - 1) > 1e-9:
                        return False
        return True
    
    # Try to find lower rank decompositions using random search with structured patterns
    # For 3x5x6, try ranks from 90 down to see if we can find valid decompositions
    
    random.seed(42)
    
    # Try a few candidate ranks below schoolbook
    # Focus on ranks that might be achievable: 85-89
    for target_rank in range(best_rank - 1, max(50, best_rank - 15), -1):
        if time.time() > deadline:
            break
        
        # Try multiple random initializations for this rank
        max_attempts = 5
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            
            # Initialize with small integers
            U = [[random.randint(-2, 2) for _ in range(target_rank)] for _ in range(n * m)]
            V = [[random.randint(-2, 2) for _ in range(target_rank)] for _ in range(m * p)]
            W = [[random.randint(-2, 2) for _ in range(range(target_rank))] for _ in range(p * n)]
            
            if verify_decomposition(U, V, W, target_rank):
                best_result = {"U": U, "V": V, "W": W}
                best_rank = target_rank
                break
        
        if best_rank < best_rank:
            break
    
    return best_result
# EVOLVE-BLOCK-END

