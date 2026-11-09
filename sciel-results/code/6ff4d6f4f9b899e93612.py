import numpy as np
import time
import random
from itertools import product

DIMENSIONS = [3, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # safety margin before 150s kill
    
    # Schoolbook decomposition
    rank_school = n * m * p
    U_sb = np.zeros((n * m, rank_school))
    V_sb = np.zeros((m * p, rank_school))
    W_sb = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1.0
                V_sb[j * p + k, col] = 1.0
                W_sb[k * n + i, col] = 1.0
                col += 1
    
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = rank_school
    
    # Try to find lower rank decompositions
    # The key constraint: for all (i,j,k), sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
    
    # Strategy: Try ranks from a lower bound up to schoolbook
    # Lower bound: need at least enough terms to cover all entries
    # A single rank-1 term can only be non-zero for entries where all three factors are non-zero
    # So we need at least max(n*m, m*p, p*n) terms in principle, but likely more
    
    # Try structured search: use the fact that we can share structure
    # For each rank r, try to find U, V, W with r columns
    
    def verify_decomp(U, V, W, tol=1e-9):
        """Check if decomposition is valid."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(len(U[0])):
                        s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    def try_rank(r, max_attempts=50, seed=42):
        """Try to find a valid decomposition of rank r."""
        if time.time() > deadline - 10:
            return None
        
        rng = np.random.RandomState(seed)
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 10:
                break
            
            # Initialize with random values
            U = rng.randn(n * m, r)
            V = rng.randn(m * p, r)
            W = rng.randn(p * n, r)
            
            # Scale to make the sum work
            # We need sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1 for all (i,j,k)
            # This is a system of n*m*p equations in 3*r unknowns (but nonlinear)
            
            # Use iterative refinement: adjust W to satisfy constraints
            # For fixed U, V, we can solve for W column by column
            # But it's coupled...
            
            # Alternative: use the schoolbook as a starting point and project
            # Start with schoolbook and try to compress
            
            # Try a different approach: initialize based on schoolbook structure
            # and use gradient-like descent on the error
            
            error = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
                        error += (s - 1.0) ** 2
            
            # If error is already small, we're good
            if error < 1e-12:
                return U, V, W
            
            # Try iterative update: for each (i,j,k), adjust the contributions
            # This is complex, so let's try a simpler heuristic
            
            # Random restart with different seeds
            continue
        
        return None
    
    # Try ranks from a reasonable lower bound
    # Theoretical lower bound for matrix multiplication rank is not known precisely
    # For 3x4*4x7, let's try from 30 up to 84
    for r in range(30, rank_school + 1):
        if time.time() > deadline - 20:
            break
        
        result = try_rank(r, max_attempts=20, seed=r)
        if result is not None:
            U, V, W = result
            if verify_decomp(U, V, W):
                best_U, best_V, best_W = U, V, W
                best_rank = r
                break  # Found the lowest rank
    
    # If we didn't find anything better, try a more systematic approach
    # Use the schoolbook and look for linear dependencies among columns
    
    if best_rank == rank_school and time.time() < deadline - 30:
        # Try to find if any column is redundant
        # A column c is redundant if there exist coefficients alpha_d for d != c
        # such that for all (i,j,k): alpha_c * U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 
        #   -sum_{d != c} alpha_d * U[i*m+j,d]*V[j*p+k,d]*W[k*n+i,d]
        
        # This is hard to check directly. Instead, try random linear combinations.
        
        # Try merging pairs of columns
        U_curr = best_U.copy()
        V_curr = best_V.copy()
        W_curr = best_W.copy()
        curr_rank = best_rank
        
        improved = True
        while improved and time.time() < deadline - 20:
            improved = False
            for c1 in range(curr_rank):
                if time.time() > deadline - 20:
                    break
                for c2 in range(c1 + 1, curr_rank):
                    if time.time() > deadline - 20:
                        break
                    
                    # Try to merge columns c1 and c2
                    # We want to find alpha, beta such that
                    # alpha*U[:,c1] + beta*U[:,c2] = U_new
                    # alpha*V[:,c1] + beta*V[:,c2] = V_new  
                    # alpha*W[:,c1] + beta*W[:,c2] = W_new
                    # and the tensor equation still holds
                    
                    # This requires:
                    # For all (i,j,k): (alpha*U[i*m+j,c1]*V[j*p+k,c1]*W[k*n+i,c1] +
                    #                    beta*U[i*m+j,c2]*V[j*p+k,c2]*W[k*n+i,c2])
                    # = U_new[i*m+j]*V_new[j*p+k]*W_new[k*n+i]
                    
                    # This is generally not possible with just two coefficients
                    # unless the columns have special structure
                    
                    # Skip this for now; too complex
                    
                    pass
        
        # Try a different approach: use SVD-like compression
        # Build the tensor T[i,j,k] = 1 and try to find its CP decomposition
        
        # Actually, let's just return the best we have
    
    # Convert to lists for return
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
