"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 5]

import numpy as np
import time
from itertools import product

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank_target = n * m * p  # 30
    start_time = time.time()
    deadline = start_time + 140  # safety margin
    
    # Try to find a decomposition of decreasing rank
    # Start with schoolbook (rank 30) as fallback
    def make_schoolbook():
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
    
    best = make_schoolbook()
    
    # Try to find lower rank decompositions
    # The tensor equation: for all i,j,k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # We need integer or half-integer entries
    
    # For each candidate rank r, try to solve
    for r in range(25, 30):  # try ranks 25-29
        if time.time() > deadline:
            break
        # This is a system of n*m*p = 30 equations with 3*r unknowns
        # Each equation is a sum of products of three variables
        # This is a polynomial system - hard to solve exactly
        
        # Try random initialization with integer/half-integer values
        # and check if it works
        for attempt in range(100):
            if time.time() > deadline:
                break
            
            # Random initialization with values in {-1, -0.5, 0, 0.5, 1}
            vals = [-1, -0.5, 0, 0.5, 1]
            U = np.random.choice(vals, size=(n*m, r))
            V = np.random.choice(vals, size=(m*p, r))
            W = np.random.choice(vals, size=(p*n, r))
            
            # Check if decomposition is valid
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        for rr in range(r):
                            s += U[i*m+j, rr] * V[j*p+k, rr] * W[k*n+i, rr]
                        if abs(s - 1.0) > 1e-9:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            if valid:
                # Convert to lists
                return {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
    
    return best
# EVOLVE-BLOCK-END
