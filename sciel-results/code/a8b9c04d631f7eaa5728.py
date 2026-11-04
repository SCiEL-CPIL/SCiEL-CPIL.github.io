"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from fractions import Fraction
import time
import itertools
import random

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm = n * m  # 15
    mp = m * p  # 35
    pn = p * n  # 21
    total_entries = n * m * p  # 105
    
    deadline = time.time() + 140  # 140s safety margin
    
    # The tensor T[i,j,k] = 1 for all valid (i,j,k)
    # We need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
    
    # Start with schoolbook as fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(nm)]
    V_school = [[0] * rank_school for _ in range(mp)]
    W_school = [[0] * rank_school for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    best_rank = rank_school
    best_U, best_V, best_W = U_school, V_school, W_school
    
    # Try to find lower rank decompositions
    # Lower bound: max(n*p, m*p, n*m) = 35
    lower_bound = max(p * n, m * p, n * m)
    
    # Try ranks from lower_bound upward
    # For each rank r, we have 71*r variables and 105 equations
    # This is underdetermined for r >= 2 (71*2=142 > 105)
    
    # Strategy: Use random initialization and solve via least squares, then round to rationals
    # Verify exactness
    
    def verify(U, V, W, r):
        """Check if decomposition is exact."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(r):
                        s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if s != 1:
                        return False
        return True
    
    def verify_float(U, V, W, r, tol=1e-9):
        """Check if decomposition is approximately exact."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(r):
                        s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    # Try to find solutions using a structured approach
    # Key idea: For each rank r, set up the bilinear system and try to solve
    
    # Approach 1: Try specific constructions
    # For M_{n,1,p}: rank = n*p (no improvement possible since m=1)
    # For M_{2,m,p}: can we do better?
    
    # Let's try a different approach: use the fact that we can factor out common terms
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # = sum_j (A[i,j]) * (B[j,k])
    
    # One trick: if we let X = A[i,j] + A[i',j] and Y = B[j,k] + B[j',k],
    # then X*Y = A[i,j]B[j,k] + A[i,j]B[j',k] + A[i',j]B[j,k] + A[i',j]B[j',k]
    # This gives us 4 products for the cost of 1, but we also need to subtract cross terms.
    
    # For now, let's try a numerical search for small ranks
    
    # Use numpy for speed
    def try_rank(r, max_iters=50):
        """Try to find a rank-r decomposition using random search."""
        if time.time() > deadline - 10:
            return None
        
        # The system: for each (i,j,k), sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
        # This is a system of 105 equations in 71*r unknowns.
        
        # Strategy: Fix W to be random, then the equations become linear in U and V?
        # No, they're bilinear in U,V for fixed W.
        
        # Alternative: Use the fact that for fixed c, the contribution is a rank-1 tensor.
        # We're trying to write the all-ones tensor as a sum of rank-1 tensors.
        
        # Let's try: initialize U, V, W randomly, then use gradient descent on the loss
        # L = sum_{i,j,k} (sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] - 1)^2
        
        # But we need exact solutions, not approximate.
        
        # For small r, let's try to solve exactly using rational arithmetic.
        # This is computationally expensive, so only for very small r.
        
        return None
    
    # Try a structured construction
    # For M_{3,5,7}, let's see if we can use a Kronecker product or similar structure.
    
    # Actually, let's just return the schoolbook for now if we can't find better quickly.
    # But let's try a few more things.
    
    # Try rank = 35 (the lower bound)
    # This is 71*35 = 2485 variables, 105 equations. Very underdetermined.
    # We can set most variables to 0 and solve for the rest.
    
    # Simple approach: set U, V, W to have sparse structure
    # For example, let U[i*m+j, c] = 1 if c == i*m+j, else 0
    # Then we need V[j*p+k, i*m+j] * W[k*n+i, i*m+j] = 1 for all j,k
    # This requires V[j*p+k, i*m+j] and W[k*n+i, i*m+j] to be nonzero for all j,k
    # But V has shape (35, r) and W has shape (21, r)
    # For fixed i, j: V[j*p+k, i*m+j] must be nonzero for all k
    # This means column i*m+j of V must have nonzero entries in rows j*p+k for all k
    # Similarly for W.
    
    # This doesn't immediately give a better construction.
    
    # Let's just return the schoolbook decomposition.
    # The task is to find rank-minimal, but without more time, schoolbook is our best bet.
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
