"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # safety margin
    
    # The tensor for matrix multiplication: T[i,j,k] = 1 if we multiply A[i,j]*B[j,k]
    # U shape: (n*m, rank) = (6, rank)
    # V shape: (m*p, rank) = (18, rank)  
    # W shape: (p*n, rank) = (12, rank)
    
    def verify(U, V, W):
        """Check if U, V, W form a valid decomposition."""
        rank = U.shape[1]
        # Compute tensor from factors
        # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(rank):
                        val += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True
    
    def is_valid_entries(U, V, W):
        """Check all entries are integers or half-integers."""
        for mat in [U, V, W]:
            for row in mat:
                for x in row:
                    if abs(x*2 - round(x*2)) > 1e-9:
                        return False
        return True
    
    # Try rank from lower bound up
    # Lower bound: max(n*m, m*p, p*n) but we can do better
    # For 2x3 x 3x6, let's try ranks starting from reasonable values
    
    best_result = None
    
    # First try the schoolbook as fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j][col] = 1
                V_school[j*p+k][col] = 1
                W_school[k*n+i][col] = 1
                col += 1
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    
    # Try to find lower rank decompositions
    # The rank of matrix multiplication tensor for m x n * n x p is known to be 
    # at most mn + np + pm - 2 for some cases, but for small cases we can search
    
    # Try various ranks
    for rank in range(max(n*m, m*p, p*n), rank_school + 1):
        if time.time() > deadline - 10:
            break
        
        # Try random search with rounding
        found = False
        attempts = 0
        max_attempts = 5000 if rank < 25 else 1000
        
        while attempts < max_attempts and time.time() < deadline - 5:
            attempts += 1
            
            # Random initialization with small values
            U = [[random.choice([-1, 0, 1, 0.5, -0.5]) for _ in range(rank)] for _ in range(n*m)]
            V = [[random.choice([-1, 0, 1, 0.5, -0.5]) for _ in range(rank)] for _ in range(m*p)]
            W = [[random.choice([-1, 0, 1, 0.5, -0.5]) for _ in range(rank)] for _ in range(p*n)]
            
            if verify(U, V, W) and is_valid_entries(U, V, W):
                best_result = {"U": U, "V": V, "W": W}
                found = True
                break
        
        if found:
            # Found a decomposition at this rank, can stop
            break
    
    return best_result
# EVOLVE-BLOCK-END
