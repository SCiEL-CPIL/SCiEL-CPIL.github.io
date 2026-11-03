"""Matrix multiplication 3x4x4 - rank minimization search."""
import numpy as np
import time
import random
from itertools import combinations

DIMENSIONS = [3, 4, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    # Build schoolbook decomposition
    U = np.zeros((n * m, schoolbook_rank))
    V = np.zeros((m * p, schoolbook_rank))
    W = np.zeros((p * n, schoolbook_rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    best_rank = schoolbook_rank
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Build the multiplication tensor T[i,j,k] = 1 if valid
    # T is a tensor where T[i,m_j,p_k] = 1
    # Decomposition: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    def verify(U_mat, V_mat, W_mat, rank):
        """Check if decomposition is exact."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    expected = 1
                    actual = 0.0
                    for r in range(rank):
                        actual += U_mat[i*m+j, r] * V_mat[j*p+k, r] * W_mat[k*n+i, r]
                    if abs(actual - expected) > 1e-6:
                        return False
        return True
    
    def verify_half_integer(U_mat, V_mat, W_mat, rank):
        """Check if entries are half-integers and decomposition is exact."""
        for mat in [U_mat, V_mat, W_mat]:
            for row in range(mat.shape[0]):
                for col_idx in range(rank):
                    val = mat[row, col_idx]
                    if abs(val * 2 - round(val * 2)) > 1e-6:
                        return False, False
        return verify(U_mat, V_mat, W_mat, rank), True
    
    # Try rank reduction by merging columns
    # Strategy: for each pair of columns, try to see if one can be eliminated
    # by absorbing its contribution into the other
    
    random.seed(42)
    
    for target_rank in range(schoolbook_rank - 1, max(30, schoolbook_rank - 20), -1):
        if time.time() > deadline - 5:
            break
        
        # Try to find a valid decomposition of target_rank
        # Use the schoolbook and zero out some columns, then adjust
        
        U_try = np.zeros((n * m, target_rank))
        V_try = np.zeros((m * p, target_rank))
        W_try = np.zeros((p * n, target_rank))
        
        # Start with schoolbook columns
        for r in range(target_rank):
            U_try[:, r] = U[:, r]
            V_try[:, r] = V[:, r]
            W_try[:, r] = W[:, r]
        
        # Try local optimization: randomly perturb and check
        found = False
        num_attempts = 50 if target_rank >= 40 else 200
        
        for attempt in range(num_attempts):
            if time.time() > deadline - 5:
                break
            
            U_opt = U_try.copy()
            V_opt = V_try.copy()
            W_opt = W_try.copy()
            
            # Random perturbation
            scale = 0.1 if attempt < 10 else 0.05
            for r in range(target_rank):
                for idx in range(n*m):
                    U_opt[idx, r] += random.gauss(0, scale)
                for idx in range(m*p):
                    V_opt[idx, r] += random.gauss(0, scale)
                for idx in range(p*n):
                    W_opt[idx, r] += random.gauss(0, scale)
            
            # Greedy zero-out: set small entries to 0
            for r in range(target_rank):
                for idx in range(n*m):
                    if abs(U_opt[idx, r]) < 0.01:
                        U_opt[idx, r] = 0
                for idx in range(m*p):
                    if abs(V_opt[idx, r]) < 0.01:
                        V_opt[idx, r] = 0
                for idx in range(p*n):
                    if abs(W_opt[idx, r]) < 0.01:
                        W_opt[idx, r] = 0
            
            # Check
            ok, half_int = verify_half_integer(U_opt, V_opt, W_opt, target_rank)
            if ok and half_int:
                # Round to half-integers
                U_opt = np.round(U_opt * 2) / 2
                V_opt = np.round(V_opt * 2) / 2
                W_opt = np.round(W_opt * 2) / 2
                ok2, _ = verify_half_integer(U_opt, V_opt, W_opt, target_rank)
                if ok2:
                    best_rank = target_rank
                    best_U, best_V, best_W = U_opt.copy(), V_opt.copy(), W_opt.copy()
                    found = True
                    break
            
            if time.time() > deadline - 5:
                break
        
        if found:
            # Try to go lower
            continue
    
    # If no better found, return schoolbook
    if best_rank == schoolbook_rank:
        # Return as lists for the interface
        U_list = [[best_U[i, j] for j in range(best_rank)] for i in range(n * m)]
        V_list = [[best_V[i, j] for j in range(best_rank)] for i in range(m * p)]
        W_list = [[best_W[i, j] for j in range(best_rank)] for i in range(p * n)]
    else:
        U_list = [[best_U[i, j] for j in range(best_rank)] for i in range(n * m)]
        V_list = [[best_V[i, j] for j in range(best_rank)] for i in range(m * p)]
        W_list = [[best_W[i, j] for j in range(best_rank)] for i in range(p * n)]
    
    # Convert to integer/half-integer representation
    # Ensure all values are multiples of 0.5
    def to_half_int_list(mat_list):
        result = []
        for row in mat_list:
            new_row = []
            for val in row:
                h = round(val * 2)
                new_row.append(h / 2.0)
            result.append(new_row)
        return result
    
    U_list = to_half_int_list(U_list)
    V_list = to_half_int_list(V_list)
    W_list = to_half_int_list(W_list)
    
    return {"U": U_list, "V": V_list, "W": W_list}
