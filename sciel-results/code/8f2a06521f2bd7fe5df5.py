"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random
from itertools import product

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # 110s to leave margin
    
    # The tensor: T[i,j,k] = 1 for all valid i,j,k
    # We need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # Schoolbook as fallback (rank 24)
    best_rank = n * m * p
    best_U = [[0] * best_rank for _ in range(n * m)]
    best_V = [[0] * best_rank for _ in range(m * p)]
    best_W = [[0] * best_rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                best_U[i * m + j][column] = 1
                best_V[j * p + k][column] = 1
                best_W[k * n + i][column] = 1
                column += 1
    
    # Strategy: Try to find a lower-rank decomposition
    # For 2x3x4, the rank must be at least max(n*m, m*p, p*n) = max(6, 12, 8) = 12
    # Let's try ranks from 12 upward
    
    # Key insight: We can use a structured approach
    # For each rank r, we have:
    # U: (n*m) x r = 6 x r
    # V: (m*p) x r = 12 x r
    # W: (p*n) x r = 8 x r
    # Total variables: 26r
    # Equations: 24
    
    # This is underdetermined for r >= 1, but we need exact solutions with integer/half-integer entries
    
    # Approach 1: Try specific structured decompositions
    # For rank 12, we might be able to use a Kronecker product structure
    
    # Let's try a more systematic search
    # For small ranks, we can try to fix some entries and solve for others
    
    # Try ranks from 12 to 23
    for rank in range(12, 24):
        if time.time() > deadline:
            break
        
        # For this rank, try to find a solution
        # Use a structured approach with random restarts
        
        # The entries should be from {-1, -0.5, 0, 0.5, 1}
        candidates = [-1, -0.5, 0, 0.5, 1]
        
        # For efficiency, use vectorized operations
        max_iters = 20000
        found = False
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # Random initialization with small values
            U = np.zeros((n*m, rank))
            V = np.zeros((m*p, rank))
            W = np.zeros((p*n, rank))
            
            # Initialize with random values from candidates
            U = np.array([[random.choice(candidates) for _ in range(rank)] for _ in range(n*m)])
            V = np.array([[random.choice(candidates) for _ in range(rank)] for _ in range(m*p)])
            W = np.array([[random.choice(candidates) for _ in range(rank)] for _ in range(p*n)])
            
            # Check if this is a solution using vectorized operations
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        # Sum over rank dimension
                        s = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                        if abs(s - 1.0) > 1e-9:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                found = True
                # Convert to list of lists
                U_list = [[U[i, r] for r in range(rank)] for i in range(n*m)]
                V_list = [[V[i, r] for r in range(rank)] for i in range(m*p)]
                W_list = [[W[i, r] for r in range(rank)] for i in range(p*n)]
                return {"U": U_list, "V": V_list, "W": W_list}
        
        if found:
            break
    
    # If we haven't found a solution, try a more sophisticated approach
    # Use the fact that we can scale the decomposition
    # If (U, V, W) is a solution, so is (c*U, V, W/c) for any non-zero c
    
    # Try to find a solution where we can normalize to get half-integers
    # For rank 12, let's try a structured approach
    
    # Actually, let's try a different strategy: use the SVD of the flattened tensor
    # But this might not give us exact integer/half-integer solutions
    
    # For now, return the best we found (schoolbook if nothing better)
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
