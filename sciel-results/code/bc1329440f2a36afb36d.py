"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random
from itertools import product

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    num_vars = n * m * p  # 24 equations
    
    # Try ranks from low to high
    # Lower bound: at least max(n*m, m*p, p*n) but that's not right
    # Actually the rank must be at least the "slice rank" lower bound
    # Let's just search from a reasonable minimum
    
    start_time = time.time()
    deadline = start_time + 110  # 110s to leave margin
    
    # The tensor: T[i,j,k] = 1 for all valid i,j,k
    # We need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # Try ranks from 8 upward (Strassen-like improvements)
    # For 2x3x4, let's try from 10 to 23
    
    best_rank = n * m * p  # schoolbook
    best_U = [[0] * best_rank for _ in range(n * m)]
    best_V = [[0] * best_rank for _ in range(m * p)]
    best_W = [[0] * best_rank for _ in range(p * n)]
    
    # Schoolbook as fallback
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                best_U[i * m + j][column] = 1
                best_V[j * p + k][column] = 1
                best_W[k * n + i][column] = 1
                column += 1
    
    # Search for lower rank
    # Use a randomized approach: try random assignments and check
    
    # For a given rank r, we have:
    # U: (n*m) x r = 6 x r
    # V: (m*p) x r = 12 x r
    # W: (p*n) x r = 8 x r
    # Total variables: 6r + 12r + 8r = 26r
    # Equations: 24
    
    # This is underdetermined for r >= 1, but we need exact solutions
    
    # Strategy: Try small rank values and use a structured search
    # The entries should be from {-1, -0.5, 0, 0.5, 1}
    
    candidates = [-1, -0.5, 0, 0.5, 1]
    
    # For efficiency, note that scaling: if (U,V,W) works, so does (cU, V, W/c) etc.
    # But we need integer/half-integer, so let's just search
    
    # Try ranks from 12 down... no, from low to high
    for rank in range(8, 24):
        if time.time() > deadline:
            break
        
        # For this rank, try to find a solution
        # Use a simple approach: fix U and V randomly, solve for W
        # Or use a more systematic approach
        
        # Actually, let's try a different approach:
        # The problem is equivalent to finding a rank-r decomposition of the tensor.
        # We can use the fact that for each (i,j,k), we need the sum to be 1.
        
        # Let's try a randomized local search
        # Start with random small values and try to optimize
        
        # Actually, for such small problems, let's try an exhaustive-ish search
        # over a restricted domain
        
        # Key insight: we can normalize. If a solution exists with rational entries,
        # we can scale to get half-integers.
        
        # Let me try a different approach: use the structure
        # For 2x3x4, there might be known decompositions.
        
        # Let's try a simple randomized search with restarts
        max_iters = 50000
        found = False
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # Random initialization with small values
            U = np.zeros((n*m, rank))
            V = np.zeros((m*p, rank))
            W = np.zeros((p*n, rank))
            
            # Initialize with random values from candidates
            for i in range(n*m):
                for r in range(rank):
                    U[i, r] = random.choice(candidates)
            for i in range(m*p):
                for r in range(rank):
                    V[i, r] = random.choice(candidates)
            for i in range(p*n):
                for r in range(rank):
                    W[i, r] = random.choice(candidates)
            
            # Check if this is a solution
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        for r in range(rank):
                            s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
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
    
    # Return best found (schoolbook if nothing better)
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
