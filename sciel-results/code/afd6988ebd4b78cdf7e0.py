"""Matrix multiplication 2,4,6 - rank-minimal exact tensor decomposition."""
import time
import numpy as np
from fractions import Fraction
from itertools import product

DIMENSIONS = [2, 4, 6]

def construct():
    deadline = time.time() + 140
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 48
    
    # Build the schoolbook decomposition as fallback
    rank = rank_schoolbook
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
    
    # Try to find a lower rank decomposition
    # The tensor T[i,j,k] = 1 for all i,j,k
    # We need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # Let's try rank 47, 46, ... down to some minimum
    # For each rank r, we have:
    # - U has n*m*r variables = 8*r
    # - V has m*p*r variables = 24*r  
    # - W has p*n*r variables = 12*r
    # Total: 44*r variables
    # Constraints: n*m*p = 48 equations (one for each i,j,k)
    
    # This is a system of 48 quadratic equations in 44*r unknowns
    # For r < 48, this is underdetermined if 44*r > 48, which is always true for r >= 2
    
    # Let's try a structured approach: fix W to be standard basis-like
    # and solve for U and V
    
    # Actually, let's try a different approach: use the fact that the tensor
    # is all-ones and try to find a decomposition with rank = n*m*p - 1 or lower
    
    # Try rank 47 first
    for test_rank in range(rank_schoolbook - 1, max(20, rank_schoolbook - 10), -1):
        if time.time() > deadline - 5:
            break
            
        # Try to find a rank-test_rank decomposition
        # Fix W[k*n+i, r] = delta_{k*n+i, r} for r < p*n = 12, and 0 otherwise
        # Then V[j*p+k, r] * W[k*n+i, r] = V[j*p+k, r] * delta_{k*n+i, r}
        # This only works if r < 12, which is too small
        
        # Better: try a random initialization and solve
        # Or use a structured decomposition
        
        # Let's try: set W[k*n+i, r] = 1 if r == k*n+i for r < 12, else 0
        # Then the equation becomes:
        # sum_{r < 12} U[i*m+j, r] * V[j*p+k, r] * delta_{k*n+i, r} = 1
        # = U[i*m+j, k*n+i] * V[j*p+k, k*n+i] = 1
        # This requires U[i*m+j, k*n+i] = V[j*p+k, k*n+i]^{-1}
        # But we need this for ALL i,j,k, so:
        # U[i*m+j, k*n+i] * V[j*p+k, k*n+i] = 1 for all i,j,k
        # This means U and V are determined, and rank = 12
        # But wait, this only works if the mapping (k,i) -> k*n+i is bijective to {0,...,11}
        # which it is since p*n = 12
        # So we'd have U[i*m+j, k*n+i] = 1/V[j*p+k, k*n+i]
        # Set V[j*p+k, k*n+i] = 1, then U[i*m+j, k*n+i] = 1
        # But U has only 8 rows and 12 columns, so U[i*m+j, col] for col in 0..11
        # For a fixed (i,j), U[i*m+j, col] must equal 1 for ALL k such that k*n+i is the column
        # But k*n+i takes different values for different k, so U[i*m+j, k*n+i] = 1 for all k
        # This means U[i*m+j, :] = [1,1,...,1] for all i,j
        # Similarly V[j*p+k, :] = [1,1,...,1] for all j,k
        # Then the sum is sum_r 1*1*delta_{k*n+i,r} = 1 (since exactly one r equals k*n+i)
        # This works! Rank = 12
        
        if test_rank >= 12:
            # Try rank 12 decomposition
            r = 12
            U = [[0] * r for _ in range(n * m)]
            V = [[0] * r for _ in range(m * p)]
            W = [[0] * r for _ in range(p * n)]
            
            # W[k*n+i, r] = 1 if r == k*n+i, else 0
            for k in range(p):
                for i in range(n):
                    W[k * n + i][k * n + i] = 1
            
            # U[i*m+j, col] = 1 for all col
            for i in range(n):
                for j in range(m):
                    for col in range(r):
                        U[i * m + j][col] = 1
            
            # V[j*p+k, col] = 1 for all col
            for j in range(m):
                for k in range(p):
                    for col in range(r):
                        V[j * p + k][col] = 1
            
            # Verify: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            # = sum_r 1 * 1 * delta_{r, k*n+i}
            # = 1 (since r = k*n+i is in range 0..11)
            # This is correct!
            
            # Verify numerically
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0
                        for r in range(r):
                            s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                        if s != 1:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook
    return {"U": U, "V": V, "W": W}

