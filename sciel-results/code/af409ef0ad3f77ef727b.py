"""Matrix multiplication 3,4,5 rank-minimal decomposition."""
import numpy as np
import time
import random
from fractions import Fraction
from itertools import product

DIMENSIONS = [3, 4, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    # Build the target tensor T[a, b, c_idx] = 1 if a=i*m+j, b=j*p+k, c_idx=k*n+i
    T = np.zeros((nm, mp, pn), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c_idx = k * n + i
                T[a, b, c_idx] = 1.0
    
    # First, get the schoolbook solution as fallback
    rank_school = n * m * p
    U_school = np.zeros((nm, rank_school))
    V_school = np.zeros((mp, rank_school))
    W_school = np.zeros((pn, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1
                V_school[j*p+k, col] = 1
                W_school[k*n+i, col] = 1
                col += 1
    best_result = (U_school, V_school, W_school)
    best_rank = rank_school
    
    # Try to find lower rank solutions
    # Strategy: For a given rank r, we need to find U (nm x r), V (mp x r), W (pn x r)
    # such that sum_c U[a,c]*V[b,c]*W[c_idx,c] = T[a,b,c_idx] for all a,b,c_idx
    
    # This is equivalent to: for each c_idx, the matrix M_c_idx = sum_c W[c_idx,c] * (U[:,c] V[:,c]^T) = T[:,:,c_idx]
    
    # We can try a structured approach:
    # 1. Start with random U, V
    # 2. For each c_idx, solve for W[c_idx,:] using least squares
    # 3. Check if the residual is small enough
    # 4. If not, try different U, V
    
    # But this is very expensive. Let's try a smarter approach.
    
    # Alternative: Use the fact that the tensor has a specific structure.
    # T[a,b,c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i
    
    # For 3x4x5, let's try ranks from 35 to 59 (since 60 is schoolbook)
    
    random.seed(42)
    np.random.seed(42)
    
    # Try a more systematic search
    # For each rank, try multiple random initializations
    
    for rank in range(35, best_rank):
        if time.time() > deadline:
            break
        
        # Try a limited number of trials per rank
        trials = 50 if rank < 50 else 20
        
        for trial in range(trials):
            if time.time() > deadline:
                break
            
            # Random initialization with small integers
            # Use a mix of 0, 1, -1 to keep values small
            U = np.random.randint(0, 2, size=(nm, rank)).astype(np.float64)
            V = np.random.randint(0, 2, size=(mp, rank)).astype(np.float64)
            
            # For each c_idx, solve for W[c_idx, :]
            # The equation is: sum_c U[a,c]*V[b,c]*W[c_idx,c] = T[a,b,c_idx]
            # This is a linear system in W[c_idx,:]
            
            W = np.zeros((pn, rank))
            valid = True
            
            for c_idx in range(pn):
                # Build the linear system matrix M where M[flat_idx, c] = U[a,c]*V[b,c]
                # and rhs[flat_idx] = T[a,b,c_idx]
                M = np.zeros((nm * mp, rank))
                rhs = np.zeros(nm * mp)
                
                for a in range(nm):
                    for b in range(mp):
                        idx = a * mp + b
                        for c in range(rank):
                            M[idx, c] = U[a, c] * V[b, c]
                        rhs[idx] = T[a, b, c_idx]
                
                # Solve M @ W[c_idx,:]^T = rhs using least squares
                try:
                    sol, residuals, rank_s, sv = np.linalg.lstsq(M, rhs, rcond=None)
                    W[c_idx, :] = sol
                    
                    # Check if residual is small
                    residual = np.linalg.norm(M @ sol - rhs)
                    if residual > 1e-8:
                        valid = False
                        break
                except:
                    valid = False
                    break
            
            if valid:
                # Verify the full decomposition
                max_err = 0
                for c_idx in range(pn):
                    prod = np.zeros((nm, mp))
                    for c in range(rank):
                        prod += U[:, c] * V[:, c] * W[c_idx, c]
                    err = np.max(np.abs(prod - T[:, :, c_idx]))
                    max_err = max(max_err, err)
                
                if max_err < 1e-8:
                    # Found a valid decomposition!
                    # Try to clean up to rational/integer entries
                    U_clean = np.round(U * 2) / 2
                    V_clean = np.round(V * 2) / 2
                    W_clean = np.round(W * 2) / 2
                    
                    # Verify cleaned version
                    max_err2 = 0
                    for c_idx in range(pn):
                        prod = np.zeros((nm, mp))
                        for c in range(rank):
                            prod += U_clean[:, c] * V_clean[:, c] * W_clean[c_idx, c]
                        err = np.max(np.abs(prod - T[:, :, c_idx]))
                        max_err2 = max(max_err2, err)
                    
                    if max_err2 < 1e-8:
                        best_result = (U_clean, V_clean, W_clean)
                        best_rank = rank
                        break  # Found lower rank, break trial loop
                    else:
                        best_result = (U, V, W)
                        best_rank = rank
                        break
            # Continue trying if not valid
    
    if time.time() > deadline:
        pass  # Return best found
    
    U, V, W = best_result
    
    # Convert to lists of lists with integer/half-integer entries
    def to_list(mat):
        result = []
        for row in mat:
            result.append([float(x) for x in row])
        return result
    
    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
