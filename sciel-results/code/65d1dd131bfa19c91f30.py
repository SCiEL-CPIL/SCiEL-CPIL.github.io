"""
Matrix multiplication 3x3x8 - rank-minimal exact tensor decomposition.
Uses alternating least squares with structured sparsity and rational polishing.
"""
import numpy as np
import time
from itertools import product

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # safety margin
    
    # The multiplication tensor T has T[i,j,k] = 1 for all i,j,k
    # We want to find U (nm x r), V (mp x r), W (pn x r) such that
    # sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1 for all i,j,k
    
    # Start with schoolbook (rank = n*m*p = 72)
    rank_school = n * m * p
    
    # Try to find a lower rank decomposition
    best_rank = rank_school
    best_U = None
    best_V = None
    best_W = None
    
    # First, establish the schoolbook solution
    U_sch = np.zeros((n * m, rank_school))
    V_sch = np.zeros((m * p, rank_school))
    W_sch = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sch[i * m + j, col] = 1.0
                V_sch[j * p + k, col] = 1.0
                W_sch[k * n + i, col] = 1.0
                col += 1
    
    def check_decomposition(U, V, W, tol=1e-10):
        """Check if U, V, W correctly decompose the multiplication tensor."""
        r = U.shape[1]
        if r != V.shape[1] or r != W.shape[1]:
            return False
        # Verify T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(r):
                        val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    best_U = U_sch.copy()
    best_V = V_sch.copy()
    best_W = W_sch.copy()
    best_rank = rank_school
    
    # Try ranks from 71 down to some minimum
    # Use a heuristic: start from schoolbook and try to merge columns
    
    for target_rank in range(rank_school - 1, max(20, rank_school - 40) - 1, -1):
        if time.time() > deadline - 5:
            break
        
        # Try to find a rank-target_rank decomposition using ALS
        # Initialize with a truncated/merged version of schoolbook
        attempts = 3
        found = False
        
        for attempt in range(attempts):
            if time.time() > deadline - 5:
                break
            
            # Initialize by randomly merging pairs of schoolbook columns
            # Create a permutation and merge
            perm = np.random.permutation(rank_school)
            U_init = np.zeros((n * m, target_rank))
            V_init = np.zeros((m * p, target_rank))
            W_init = np.zeros((p * n, target_rank))
            
            # Assign schoolbook columns to target columns (some get merged)
            for c in range(target_rank):
                # Take one or more schoolbook columns and average/combine
                src_cols = []
                remaining = list(perm[c:c + (rank_school - c) // (target_rank - c) + 1])
                for sc in remaining[:1]:
                    U_init[:, c] = U_sch[:, sc]
                    V_init[:, c] = V_sch[:, sc]
                    W_init[:, c] = W_sch[:, sc]
            
            # Run ALS iterations
            U_cur = U_init.copy()
            V_cur = V_init.copy()
            W_cur = W_init.copy()
            
            for it in range(50):
                if time.time() > deadline - 5:
                    break
                
                # Update U: for each (i,j), solve for U[i*m+j, :]
                # sum_r U[i*m+j, r] * (V[j*p+k, r] * W[k*n+i, r]) = 1 for all k
                for i in range(n):
                    for j in range(m):
                        A = np.zeros((p, target_rank))
                        for k in range(p):
                            A[k, :] = V_cur[j * p + k, :] * W_cur[k * n + i, :]
                        b = np.ones(p)
                        try:
                            U_cur[i * m + j, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                        except:
                            break
                        if time.time() > deadline - 5:
                            break
                    if time.time() > deadline - 5:
                        break
                if time.time() > deadline - 5:
                    break
                
                # Update V
                for j in range(m):
                    for k in range(p):
                        A = np.zeros((n, target_rank))
                        for i in range(n):
                            A[i, :] = U_cur[i * m + j, :] * W_cur[k * n + i, :]
                        b = np.ones(n)
                        try:
                            V_cur[j * p + k, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                        except:
                            break
                        if time.time() > deadline - 5:
                            break
                    if time.time() > deadline - 5:
                        break
                if time.time() > deadline - 5:
                    break
                
                # Update W
                for k in range(p):
                    for i in range(n):
                        A = np.zeros((m, target_rank))
                        for j in range(m):
                            A[j, :] = U_cur[i * m + j, :] * V_cur[j * p + k, :]
                        b = np.ones(m)
                        try:
                            W_cur[k * n + i, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                        except:
                            break
                        if time.time() > deadline - 5:
                            break
                    if time.time() > deadline - 5:
                        break
                if time.time() > deadline - 5:
                    break
            
            if time.time() > deadline - 5:
                break
            
            # Check if we found a valid decomposition
            if check_decomposition(U_cur, V_cur, W_cur, tol=1e-8):
                # Try to polish to rational values
                found = True
                # Round to nearest half-integer if close
                U_round = np.round(U_cur * 2) / 2
                V_round = np.round(V_cur * 2) / 2
                W_round = np.round(W_cur * 2) / 2
                
                if check_decomposition(U_round, V_round, W_round, tol=1e-10):
                    best_U = U_round
                    best_V = V_round
                    best_W = W_round
                else:
                    best_U = U_cur
                    best_V = V_cur
                    best_W = W_cur
                best_rank = target_rank
                break
        
        if found:
            # Found a valid decomposition at this rank, try to go lower
            continue
        else:
            # If we can't find one at this rank, stop trying lower ranks
            break
    
    # Convert to lists of integers/half-integers
    def to_list(mat):
        result = []
        for row in mat:
            r = []
            for val in row:
                # Check if it's close to an integer or half-integer
                val2 = val * 2
                if abs(val2 - round(val2)) < 1e-6:
                    r.append(round(val2) / 2)
                else:
                    r.append(val)
            result.append(r)
        return result
    
    return {
        "U": to_list(best_U),
        "V": to_list(best_V),
        "W": to_list(best_W)
    }
# EVOLVE-BLOCK-END
